import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re

# --- ページ設定 ---
st.set_page_config(page_title="Honda 競合・流出分析ダッシュボード", layout="wide")

# --- 1. データ読み込み & 正規化前処理 ---
@st.cache_data
def load_data():
    csv_path = "data/input/SV/NVES_Honda_Analysis_Cloud.csv"
    data = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    brand_mapping = {'Mercedes': 'Mercedes-Benz', 'Vinfast': 'VinFast'}
    data['Brand (Disposed)'] = data['Brand (Disposed)'].replace(brand_mapping)
    data['New Model Purchased - Brand'] = data['New Model Purchased - Brand'].replace(brand_mapping)
    
    # 数値変換とウェイトの欠損埋め（0）
    data['Price_Num'] = pd.to_numeric(data['Purchase Price (Detailed)'], errors='coerce')
    data['Source of Sales Weight'] = pd.to_numeric(data['Source of Sales Weight'], errors='coerce').fillna(0)
    data['Repurchase Loyalty Weight'] = pd.to_numeric(data['Repurchase Loyalty Weight'], errors='coerce').fillna(0)
    
    def clean_model_name(brand, model):
        brand, model = str(brand).strip(), str(model).strip()
        full = model if model.lower().startswith(brand.lower()) else f"{brand} {model}"
        words = full.split()
        if len(words) >= 2:
            if words[1].lower() == 'model' and len(words) >= 3:
                return f"{words[0]} {words[1]} {words[2]}"
            return f"{words[0]} {words[1]}"
        return full

    data['Clean_Model_Prev'] = data.apply(lambda x: clean_model_name(x['Brand (Disposed)'], x['Model (Disposed)']), axis=1)
    data['Clean_Model_Curr'] = data.apply(lambda x: clean_model_name(x['New Model Purchased - Brand'], x['New Model Purchased - Make/Model/Series (Alpha Order)']), axis=1)
    return data

@st.cache_data
def load_jato_specs():
    """JATOデータから車両スペック情報を読み込み"""
    try:
        jato_path = "data/input/JATO_USA_MMix.csv"
        jato = pd.read_csv(jato_path, encoding='shift-jis')
        
        # カラム名を取得（エンコーディング問題回避）
        maker_col = jato.columns[0]  # メーカー
        model_col = jato.columns[1]  # 車名
        body_col = jato.columns[3]   # BodyType
        seat_col = jato.columns[13]  # 乗車定員
        price_col = jato.columns[10] # 本体価格
        
        # 最新年のデータのみ使用
        jato_latest = jato[jato['SalesYear'] == jato['SalesYear'].max()].copy()
        
        # モデル名を統一（大文字小文字を正規化）
        jato_latest['Model_Key'] = jato_latest[maker_col].str.title() + ' ' + jato_latest[model_col].str.title()
        
        # 集約（同じモデルで複数グレードがある場合は平均）
        specs = jato_latest.groupby('Model_Key').agg({
            body_col: 'first',
            seat_col: 'first',
            price_col: 'mean',
            'Segment': 'first'
        }).reset_index()
        
        # カラム名を英語に変更
        specs.columns = ['Model_Key', 'BodyType', 'Seating', 'Price', 'Segment']
        
        return specs
    except Exception as e:
        st.warning(f"JATOデータ読み込みエラー: {e}")
        return pd.DataFrame()


try:
    df = load_data()
    jato_specs = load_jato_specs()
except Exception as e:
    st.error(f"読み込み失敗: {e}")
    st.stop()

# --- 2. 基準価格の算出 ---
target_honda_models = ['Civic', 'CR-V', 'HR-V', 'Accord', 'Odyssey']
honda_family = ['Honda', 'Acura']
model_benchmarks = {m: df[(df['New Model Purchased - Brand'] == 'Honda') & (df['New Model Purchased - Make/Model/Series (Alpha Order)'].str.contains(m, case=False, na=False))]['Price_Num'].mean() for m in target_honda_models}

# --- サイドバー設定 ---
st.sidebar.header("📊 分析ターゲット設定")
mode = st.sidebar.radio("集計基準", ["ウェイトバック (Market)", "生値 (Raw)"])

# 重みの使い分け設定
if mode == "ウェイトバック (Market)":
    w_in = 'Source of Sales Weight'
    w_out = 'Repurchase Loyalty Weight'
    y_label = "市場ボリューム (Weighted)"
else:
    # 生値の場合は RECORD_ID の個数をカウントするためにダミーで1を振る
    df['ones'] = 1
    w_in = 'ones'
    w_out = 'ones'
    y_label = "サンプル数 (Raw)"

selected_honda = st.sidebar.selectbox("分析対象のHondaモデルを選択", target_honda_models)

# --- 描画関数：価格ヒストグラム ---
def draw_price_histogram(data, title, weight_col, active_model=None):
    data_sorted = data.sort_values('Status', ascending=False)
    fig = px.histogram(data_sorted, x="Price_Num", y=weight_col, histfunc="sum",
                       color="Status", barmode="overlay", title=title,
                       color_discrete_map={'Stay (Honda/Acura)':'#2ecc71', 'Outflow (Competitors)':'#e67e22'},
                       labels={'Price_Num': '購入価格 (USD)', weight_col: y_label}, opacity=0.6)
    
    for m_name, m_price in model_benchmarks.items():
        if pd.isna(m_price): continue
        is_active = (m_name == active_model)
        color = "#FF0000" if is_active else "#95a5a6"
        fig.add_vline(x=m_price, line_dash="solid" if is_active else "dash", line_color=color, line_width=3 if is_active else 1,
                      annotation_text=m_name, annotation_position="top left", annotation_font_color=color)
    fig.update_layout(yaxis_title=y_label)
    return fig

# --- 描画関数：横棒グラフ（価格情報付き） ---
def draw_h_bar_with_price(series, price_series, title, color):
    """
    series: 集計値（ウェイトまたはカウント）
    price_series: 平均価格
    """
    calculated_height = 400 + (len(series) * 20)
    
    # データフレーム化
    df_plot = pd.DataFrame({
        'value': series.values,
        'avg_price': price_series.reindex(series.index).values
    }, index=series.index)
    
    # ラベルに価格情報を追加
    df_plot['label'] = df_plot.index + ' ($' + df_plot['avg_price'].apply(lambda x: f'{x:,.0f}' if pd.notna(x) else 'N/A') + ')'
    
    fig = px.bar(df_plot, x='value', y='label', orientation='h', 
                 color_discrete_sequence=[color], title=title,
                 hover_data={'avg_price': ':.0f'})
    fig.update_layout(height=calculated_height, margin=dict(l=280, r=20, t=50, b=50),
                      yaxis=dict(title="", autorange="reversed"), 
                      xaxis=dict(title=y_label), showlegend=False)
    return fig

# --- 描画関数：横棒グラフ（シンプル版） ---
def draw_h_bar(series, title, color):
    calculated_height = 400 + (len(series) * 20)
    fig = px.bar(series, orientation='h', color_discrete_sequence=[color], title=title)
    fig.update_layout(height=calculated_height, margin=dict(l=220, r=20, t=50, b=50),
                      yaxis=dict(title="", autorange="reversed"), xaxis=dict(title=y_label), showlegend=False)
    return fig

# --- メイン ---
st.title("🚗 Honda マーケット分析ダッシュボード")
tab_overall, tab_specific, tab_compare = st.tabs(["📊 Honda全体分析", "🔍 個別モデル深掘り", "⚔️ 競合比較"])

with tab_overall:
    st.header("Hondaブランド全体の流入・流出構造")
    h_all_dis = df[df['Brand (Disposed)'] == 'Honda'].copy()
    h_all_dis['Status'] = h_all_dis['New Model Purchased - Brand'].apply(lambda x: 'Stay (Honda/Acura)' if x in honda_family else 'Outflow (Competitors)')
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("全体離反率")
        v = h_all_dis.groupby('Status')[w_out].sum()
        st.plotly_chart(px.pie(values=v.values, names=v.index, hole=0.4, color_discrete_map={'Stay (Honda/Acura)':'#2ecc71', 'Outflow (Competitors)':'#e74c3c'}))
    with c2:
        st.subheader("価格移動実態")
        st.plotly_chart(draw_price_histogram(h_all_dis, "Honda全処分者の価格移動", w_out))

    st.divider()
    cin, cout = st.columns(2)
    with cin:
        st.subheader("📥 流入分析")
        in_df_all = df[(df['New Model Purchased - Brand'] == 'Honda') & (~df['Brand (Disposed)'].isin(honda_family))]
        in_brand = in_df_all.groupby('Brand (Disposed)')[w_in].sum().sort_values(ascending=False).head(15)
        in_brand_price = in_df_all.groupby('Brand (Disposed)')['Price_Num'].mean()
        st.plotly_chart(draw_h_bar_with_price(in_brand, in_brand_price, "流入元ブランド TOP15 (平均価格)", '#3498db'))
    with cout:
        st.subheader("📤 流出分析")
        out_df_all = h_all_dis[h_all_dis['Status'] == 'Outflow (Competitors)']
        out_brand = out_df_all.groupby('New Model Purchased - Brand')[w_out].sum().sort_values(ascending=False).head(15)
        out_brand_price = out_df_all.groupby('New Model Purchased - Brand')['Price_Num'].mean()
        st.plotly_chart(draw_h_bar_with_price(out_brand, out_brand_price, "流出先ブランド TOP15 (平均価格)", '#e74c3c'))

with tab_specific:
    st.header(f"Honda {selected_honda} インサイト")
    m_df = df[(df['Brand (Disposed)'] == 'Honda') & (df['Model (Disposed)'].str.contains(selected_honda, case=False, na=False))].copy()
    m_df['Status'] = m_df['New Model Purchased - Brand'].apply(lambda x: 'Stay (Honda/Acura)' if x in honda_family else 'Outflow (Competitors)')

    c1, c2 = st.columns([1, 2])
    with c1:
        v = m_df.groupby('Status')[w_out].sum()
        st.plotly_chart(px.pie(values=v.values, names=v.index, hole=0.4, color_discrete_map={'Stay (Honda/Acura)':'#2ecc71', 'Outflow (Competitors)':'#e74c3c'}))
    with c2:
        st.plotly_chart(draw_price_histogram(m_df, f"{selected_honda} 価格移動", w_out, active_model=selected_honda))

    st.divider()
    top_n = st.slider("表示件数", 10, 50, 20, 5)
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("📥 流入元モデル分析")
        in_m = df[(df['New Model Purchased - Make/Model/Series (Alpha Order)'].str.contains(selected_honda, case=False, na=False)) & (~df['Brand (Disposed)'].isin(honda_family + ['Did Not Dispose', 'Did Not Own', 'Did not own', 'Did not dispose']))]
        in_stats = in_m.groupby('Clean_Model_Prev')[w_in].sum().sort_values(ascending=False).head(top_n)
        in_price = in_m.groupby('Clean_Model_Prev')['Price_Num'].mean()
        st.plotly_chart(draw_h_bar_with_price(in_stats, in_price, f"流入元モデル TOP{top_n} (平均価格)", '#3498db'))
    with c4:
        st.subheader("📤 流出先モデル分析")
        out_m_df = m_df[m_df['Status'] == 'Outflow (Competitors)']
        out_stats = out_m_df.groupby('Clean_Model_Curr')[w_out].sum().sort_values(ascending=False).head(top_n)
        out_price = out_m_df.groupby('Clean_Model_Curr')['Price_Num'].mean()
        st.plotly_chart(draw_h_bar_with_price(out_stats, out_price, f"逃げ先モデル TOP{top_n} (平均価格)", '#ec7063'))

with tab_compare:
    st.header("⚔️ 競合モデル比較")
    st.markdown("Hondaモデルと主要競合モデルのスペック・価格を比較します")
    
    # 競合モデルリストの取得
    m_df_comp = df[(df['Brand (Disposed)'] == 'Honda') & 
                   (df['Model (Disposed)'].str.contains(selected_honda, case=False, na=False))].copy()
    m_df_comp['Status'] = m_df_comp['New Model Purchased - Brand'].apply(
        lambda x: 'Stay' if x in honda_family else 'Outflow')
    
    competitors_df = m_df_comp[m_df_comp['Status'] == 'Outflow']
    top_competitors = competitors_df.groupby('Clean_Model_Curr')[w_out].sum().sort_values(ascending=False).head(20)
    
    if len(top_competitors) == 0:
        st.warning(f"{selected_honda}からの流出データがありません")
        st.stop()
    
    col_select1, col_select2 = st.columns(2)
    with col_select1:
        honda_model_full = f"Honda {selected_honda}"
        st.info(f"**分析対象**: {honda_model_full}")
    
    with col_select2:
        competitor_model = st.selectbox(
            "比較する競合モデルを選択",
            options=top_competitors.index.tolist(),
            help="流出先TOP20から選択"
        )
    
    st.divider()
    
    # スペック比較表示
    col1, col2 = st.columns(2)
    
    def display_model_card(model_name, col, data_source, is_honda=False):
        """モデルカードを表示"""
        with col:
            st.subheader(f"🚗 {model_name}")
            
            # SVデータから価格情報取得
            if is_honda:
                model_data = data_source
            else:
                model_data = data_source[data_source['Clean_Model_Curr'] == model_name]
            
            col_a, col_b = st.columns(2)
            
            if len(model_data) > 0:
                avg_price = model_data['Price_Num'].mean()
                
                # セグメント情報
                if 'New Model Segment' in model_data.columns:
                    segments = model_data['New Model Segment'].dropna()
                    segment = segments.mode()[0] if len(segments) > 0 else 'N/A'
                else:
                    segment = 'N/A'
                
                with col_a:
                    st.metric("平均購入価格 (SV調査)", f"${avg_price:,.0f}" if pd.notna(avg_price) else "N/A")
                with col_b:
                    st.metric("セグメント", segment)
            else:
                st.warning("SVデータなし")
            
            # JATOデータからスペック取得
            if len(jato_specs) > 0:
                # モデル名のマッチング（柔軟に）
                # ブランド名を除去してモデル名のみで検索
                search_parts = model_name.split()
                if len(search_parts) >= 2:
                    search_name = search_parts[1]  # 2番目の単語（モデル名）
                else:
                    search_name = model_name
                
                jato_match = jato_specs[jato_specs['Model_Key'].str.contains(search_name, case=False, na=False)]
                
                if len(jato_match) > 0:
                    spec = jato_match.iloc[0]
                    
                    st.divider()
                    st.markdown("**📋 JATOスペック情報**")
                    
                    col_c, col_d, col_e = st.columns(3)
                    with col_c:
                        st.metric("ボディタイプ", spec['BodyType'] if pd.notna(spec['BodyType']) else 'N/A')
                    with col_d:
                        st.metric("乗車定員", f"{int(spec['Seating'])}人" if pd.notna(spec['Seating']) else 'N/A')
                    with col_e:
                        if pd.notna(spec['Price']):
                            st.metric("JATO価格 (USA)", f"${spec['Price']:,.0f}")
                        else:
                            st.metric("JATO価格", "N/A")
                else:
                    st.info(f"JATOスペック情報なし (検索: {search_name})")
            else:
                st.info("JATOデータ未読み込み")
    
    # Honda側のデータ
    honda_data = df[(df['New Model Purchased - Brand'] == 'Honda') & 
                    (df['New Model Purchased - Make/Model/Series (Alpha Order)'].str.contains(selected_honda, case=False, na=False))]
    competitor_data = df.copy()
    
    display_model_card(honda_model_full, col1, honda_data, is_honda=True)
    display_model_card(competitor_model, col2, competitor_data, is_honda=False)
    
    # 流出ボリューム表示
    st.divider()
    st.subheader("📊 流出ボリューム分析")
    outflow_volume = competitors_df[competitors_df['Clean_Model_Curr'] == competitor_model][w_out].sum()
    total_outflow = competitors_df[w_out].sum()
    
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        st.metric("この競合への流出", f"{outflow_volume:,.1f}")
    with col_v2:
        st.metric("全流出", f"{total_outflow:,.1f}")
    with col_v3:
        share = (outflow_volume / total_outflow * 100) if total_outflow > 0 else 0
        st.metric("流出シェア", f"{share:.1f}%")
    
    # 価格比較チャート
    st.divider()
    st.subheader("💰 価格帯比較")
    
    # Honda vs 競合の価格分布
    honda_prices = honda_data['Price_Num'].dropna()
    comp_prices = competitor_data[competitor_data['Clean_Model_Curr'] == competitor_model]['Price_Num'].dropna()
    
    if len(honda_prices) > 0 and len(comp_prices) > 0:
        price_comparison = pd.DataFrame({
            'Price': list(honda_prices) + list(comp_prices),
            'Model': [honda_model_full] * len(honda_prices) + [competitor_model] * len(comp_prices)
        })
        
        fig_price_comp = px.histogram(price_comparison, x='Price', color='Model', 
                                      barmode='overlay', opacity=0.7,
                                      title="価格分布比較",
                                      labels={'Price': '購入価格 (USD)', 'count': '件数'},
                                      color_discrete_map={honda_model_full: '#2ecc71', competitor_model: '#e74c3c'})
        st.plotly_chart(fig_price_comp, use_container_width=True)
    else:
        st.info("価格比較データが不足しています")
