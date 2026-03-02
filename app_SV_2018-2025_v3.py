import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ページ基本設定
st.set_page_config(layout="wide", page_title="Honda vs Toyota 2018-2025 Evolution")

# データの読み込み
@st.cache_data
def load_master_data():
    df = pd.read_csv("SV_Streamlit_Master_new.csv", low_memory=False)
    df['Year'] = df['Year'].astype(str)
    
    # Is_Major_Modelを作成
    honda_major = ['Civic', 'CR-V', 'Accord', 'Pilot', 'Passport']
    toyota_major = ['Corolla', 'RAV4', 'Camry', 'Highlander', 'Land Cruiser', 'Tundra', 'Sequoia', 'Grand Highlander']
    
    def is_major_model(model_name):
        if pd.isna(model_name):
            return False
        model_str = str(model_name)
        for major in honda_major + toyota_major:
            if major in model_str:
                return True
        return False
    
    df['Is_Major_Model'] = df['Model_P'].apply(is_major_model)
    
    return df

df = load_master_data()

# 年収のソート順定義を関数外に移動
income_order = [
        "Under $15,000", "$15,000 to $19,999", "$20,000 to $24,999", 
        "$25,000 to $29,999", "$30,000 to $34,999", "$35,000 to $39,999",
        "$40,000 to $44,999", "$45,000 to $49,999", "$50,000 to $54,999",
        "$55,000 to $59,999", "$60,000 to $64,999", "$65,000 to $69,999",
        "$70,000 to $74,999", "$75,000 to $79,999", "$80,000 to $84,999",
        "$85,000 to $89,999", "$90,000 to $94,999", "$95,000 to $99,999",
        "$100,000 to $104,999", "$125,000 to $149,999", "$150,000 to $174,999",
        "$175,000 to $199,999", "$200,000 to $249,999", "$250,000 to $299,999",
        "$300,000 to $349,999", "$350,000 to $399,999", "$400,000 to $449,999",
        "$450,000 to $499,999", "$500,000 or Over"
]

# 主要モデル定義
HONDA_MAJOR = ['Civic', 'CR-V', 'Accord', 'Pilot', 'Passport', 'Odyssey', 'HR-V', 'Ridgeline', 'Insight', 'Fit', 'Element']
TOYOTA_MAJOR = ['Corolla', 'RAV4', 'Camry', 'Highlander', 'Land Cruiser', 'Tundra', 'Sequoia', 'Grand Highlander',
                'Tacoma', 'Sienna', 'Prius', '4Runner', 'Avalon', 'Venza', 'C-HR', 'GR86', 'Supra']

# --- サイドバー ---
st.sidebar.title("🔍 分析設定")

# 分析モード選択
analysis_mode = st.sidebar.radio(
    "分析モード",
    ["🏢 ブランド全体比較", "🚗 モデル別分析", "⚔️ モデル間比較"]
)

# ウェイトバック設定
st.sidebar.divider()
st.sidebar.markdown("### 📊 ウェイトバック設定")

# ウェイト列の定義
WEIGHT_DISTRIBUTION = 'A - Part 1/Paper/Abridged/Non-Response'  # 分布分析用
WEIGHT_LOYALTY = 'Repurchase Loyalty Weight'  # 流出分析用
WEIGHT_SALES = 'Source of Sales Weight'  # 販売ソース分析用

st.sidebar.info("""
**自動ウェイト適用**:
- 📊 分布分析（顧客属性・地域など）
  → `A - Part 1/Paper/Abridged/Non-Response`
  
- 🔄 流出分析（離反率・流入流出）
  → `Repurchase Loyalty Weight`
  
各分析に最適なウェイトが自動的に適用されます。
""")

# 指標計算用ヘルパー
def get_weighted_share(data, group_cols, target_col, weight_col=None):
    """ウェイト付きシェア計算（NaN値を明示的に除外）
    
    weight_colが指定されない場合は、WEIGHT_DISTRIBUTION（分布分析用）を使用
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    # デフォルトで分布分析用ウェイトを使用
    if weight_col is None:
        weight_col = WEIGHT_DISTRIBUTION
    
    # target_colのNaN値を除外
    valid_data = data[data[target_col].notna()].copy()
    
    # ウェイトバック値で集計
    stats = valid_data.groupby(group_cols + [target_col])[weight_col].sum().reset_index()
    totals = stats.groupby(group_cols)[weight_col].transform('sum')
    stats['Share (%)'] = (stats[weight_col] / totals) * 100
    stats.rename(columns={weight_col: 'Value'}, inplace=True)
    
    return stats

def create_comparison_chart(data, x_col, y_col, color_col, title, chart_type='bar'):
    """比較チャート作成"""
    if chart_type == 'bar':
        fig = px.bar(data, x=x_col, y=y_col, color=color_col, barmode='group',
                     title=title, text=y_col)
        fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    else:
        fig = px.line(data, x=x_col, y=y_col, color=color_col, markers=True,
                      title=title)
    return fig

# 年収グループ化関数（マトリクスと同じレンジを使用）
def group_income_for_matrix(income_val):
    """年収を8つのグループに分類（マトリクス用）"""
    if pd.isna(income_val):
        return None
    income_str = str(income_val)
    
    # <$50k
    if '$20,000 or Less' in income_str or '$15,001 To $20,000' in income_str:
        return '<$50k'
    elif any(x in income_str for x in ['$20,001', '$25,001', '$30,001', '$35,001', '$40,001', '$45,001']):
        return '<$50k'
    # $50-99k
    elif any(x in income_str for x in ['$50,001', '$55,001', '$60,001', '$65,001', '$70,001', '$75,001', '$80,001', '$90,001']):
        return '$50-99k'
    # $100-149k
    elif any(x in income_str for x in ['$100,001', '$125,001']):
        return '$100-149k'
    # $150-199k
    elif any(x in income_str for x in ['$150,001', '$175,001']):
        return '$150-199k'
    # $200-299k
    elif '$200,001 To $300,000' in income_str:
        return '$200-299k'
    # $300-399k
    elif '$300,001 To $400,000' in income_str:
        return '$300-399k'
    # $400-499k
    elif '$400,001 To $500,000' in income_str:
        return '$400-499k'
    # $500k+
    elif any(x in income_str for x in ['$500,000', '$500,001', '$750,000', '$750,001', '$1,000,000', 'Over']):
        return '$500k+'
    else:
        return income_str

# ==================== モード1: ブランド全体比較 ====================
if analysis_mode == "🏢 ブランド全体比較":
    st.title("🏢 Honda vs Toyota ブランド全体比較 (2018 vs 2025)")
    
    # ブランド全体比較: すべてのHonda/Toyota車を対象
    # Brand_P（購入）とBrand_D（前有車）の両方でHonda/Toyotaを含める
    # これにより、流入・残留・流出すべてのケースをカバー
    
    # Honda関連: HondaまたはAcuraを購入 OR HondaまたはAcuraから流出（全モデル）
    honda_purchase = df[df['Brand_P'].isin(['Honda', 'Acura'])].copy()
    honda_disposed = df[df['Brand_D'].isin(['Honda', 'Acura'])].copy()
    
    # Toyota関連: Toyotaを購入 OR Toyotaから流出（全モデル）
    toyota_purchase = df[df['Brand_P'] == 'Toyota'].copy()
    toyota_disposed = df[df['Brand_D'] == 'Toyota'].copy()
    
    # 結合して重複を削除
    honda_df = pd.concat([honda_purchase, honda_disposed]).drop_duplicates()
    toyota_df = pd.concat([toyota_purchase, toyota_disposed]).drop_duplicates()
    combined_df = pd.concat([honda_df, toyota_df]).drop_duplicates()
    
    # デバッグ情報
    st.sidebar.info(f"Honda: {len(honda_df)} 件, Toyota: {len(toyota_df)} 件, 合計: {len(combined_df)} 件")
    
    # タブ構成
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 全体サマリー", "👤 顧客属性", "🗺 地域分布", "💰 支払方法", "❓ 離反理由"])
    
    with tab1:
        st.subheader("ブランド全体の変化")
        
        # 離反率（Defection Rate）
        st.markdown("### 🔄 離反率（Defection Rate）")
        st.caption("前有車が同じブランドから他ブランドに流出した割合")
        st.info(f"💡 離反率分析には **{WEIGHT_LOYALTY}** を使用")
        
        # Detailed_Statusを使って離反率を計算
        # Honda: "Stay (Honda)" vs "Defection (from Honda)"
        # Toyota: "Stay (Toyota)" vs "Defection (from Toyota)"
        defection_data = []
        for brand in ['Honda', 'Toyota']:
            for year in ['2018', '2025']:
                brand_year_df = combined_df[(combined_df['Brand_D'] == brand) & (combined_df['Year'] == year)]
                if len(brand_year_df) > 0:
                    # 離反率分析にはRepurchase Loyalty Weightを使用
                    total_value = brand_year_df[WEIGHT_LOYALTY].sum()
                    defection_value = brand_year_df[brand_year_df['Detailed_Status'].str.contains(f'Defection \\(from {brand}\\)', case=False, na=False)][WEIGHT_LOYALTY].sum()
                    
                    defection_rate = (defection_value / total_value * 100) if total_value > 0 else 0
                    defection_data.append({
                        'Brand': brand,
                        'Year': year,
                        'Defection_Rate': defection_rate
                    })
        
        defection_df = pd.DataFrame(defection_data)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if len(defection_df) > 0:
                fig_def = px.bar(defection_df, x='Brand', y='Defection_Rate', color='Year',
                                barmode='group', text='Defection_Rate',
                                color_discrete_map={'2018': '#95a5a6', '2025': '#e74c3c'},
                                title="離反率の比較")
                fig_def.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                fig_def.update_layout(xaxis_title="ブランド", yaxis_title="離反率 (%)", height=400)
                st.plotly_chart(fig_def, use_container_width=True)
        
        with col2:
            st.markdown("#### 変化量")
            for brand in ['Honda', 'Toyota']:
                brand_data = defection_df[defection_df['Brand'] == brand]
                if len(brand_data) == 2:
                    val_2018 = brand_data[brand_data['Year'] == '2018']['Defection_Rate'].values[0]
                    val_2025 = brand_data[brand_data['Year'] == '2025']['Defection_Rate'].values[0]
                    change = val_2025 - val_2018
                    st.metric(f"**{brand}**", f"{val_2025:.1f}%",
                             delta=f"{change:+.1f}%",
                             delta_color="inverse")
                    st.caption(f"2018: {val_2018:.1f}%")
        
        st.divider()
        
        # 流入元ブランド
        st.markdown("### 📥 流入元ブランド TOP15")
        st.caption("他ブランドからHonda/Toyotaに流入してきた元ブランド")
        st.info(f"💡 流入分析には **{WEIGHT_LOYALTY}** を使用")
        
        # 流入元（前有車ブランド）の分析
        inflow_df = combined_df[~combined_df['Brand_D'].isin(['Did Not Dispose', 'Did Not Own', 'Did not own', 'Did not dispose'])].copy()
        
        col_h, col_t = st.columns(2)
        
        with col_h:
            st.markdown("#### 🔵 Honda への流入")
            honda_inflow = inflow_df[inflow_df['Brand_P'] == 'Honda'].groupby(['Year', 'Brand_D'])[WEIGHT_LOYALTY].sum().reset_index()
            honda_inflow.rename(columns={WEIGHT_LOYALTY: 'Value'}, inplace=True)
            
            for year in ['2018', '2025']:
                year_data = honda_inflow[honda_inflow['Year'] == year].nlargest(15, 'Value')
                
                fig_h = px.bar(year_data, x='Value', y='Brand_D', orientation='h',
                              title=f"Honda 流入元 ({year}年)",
                              color_discrete_sequence=['#3498db' if year == '2018' else '#e74c3c'])
                fig_h.update_layout(height=500, yaxis={'categoryorder':'total ascending'},
                                   xaxis_title="ボリューム", yaxis_title="")
                st.plotly_chart(fig_h, use_container_width=True)
        
        with col_t:
            st.markdown("#### 🔴 Toyota への流入")
            toyota_inflow = inflow_df[inflow_df['Brand_P'] == 'Toyota'].groupby(['Year', 'Brand_D'])[WEIGHT_LOYALTY].sum().reset_index()
            toyota_inflow.rename(columns={WEIGHT_LOYALTY: 'Value'}, inplace=True)
            
            for year in ['2018', '2025']:
                year_data = toyota_inflow[toyota_inflow['Year'] == year].nlargest(15, 'Value')
                
                fig_t = px.bar(year_data, x='Value', y='Brand_D', orientation='h',
                              title=f"Toyota 流入元 ({year}年)",
                              color_discrete_sequence=['#3498db' if year == '2018' else '#e74c3c'])
                fig_t.update_layout(height=500, yaxis={'categoryorder':'total ascending'},
                                   xaxis_title="ボリューム", yaxis_title="")
                st.plotly_chart(fig_t, use_container_width=True)
        
        st.divider()
        
        # 流出先ブランド
        st.markdown("### 📤 流出先ブランド TOP15")
        st.caption("Honda/Toyotaから他ブランドに流出した先")
        st.info(f"💡 流出分析には **{WEIGHT_LOYALTY}** を使用")
        
        # Detailed_Statusを使って離反者（流出）を抽出
        outflow_df = combined_df[combined_df['Detailed_Status'].str.contains('Defection', case=False, na=False)].copy()
        
        col_h2, col_t2 = st.columns(2)
        
        with col_h2:
            st.markdown("#### 🔵 Honda からの流出")
            honda_outflow = outflow_df[outflow_df['Brand_D'] == 'Honda'].groupby(['Year', 'Brand_P'])[WEIGHT_LOYALTY].sum().reset_index()
            honda_outflow.rename(columns={WEIGHT_LOYALTY: 'Value'}, inplace=True)
            
            for year in ['2018', '2025']:
                year_data = honda_outflow[honda_outflow['Year'] == year].nlargest(15, 'Value')
                
                fig_ho = px.bar(year_data, x='Value', y='Brand_P', orientation='h',
                               title=f"Honda 流出先 ({year}年)",
                               color_discrete_sequence=['#3498db' if year == '2018' else '#e74c3c'])
                fig_ho.update_layout(height=500, yaxis={'categoryorder':'total ascending'},
                                    xaxis_title="ボリューム", yaxis_title="")
                st.plotly_chart(fig_ho, use_container_width=True)
        
        with col_t2:
            st.markdown("#### 🔴 Toyota からの流出")
            toyota_outflow = outflow_df[outflow_df['Brand_D'] == 'Toyota'].groupby(['Year', 'Brand_P'])[WEIGHT_LOYALTY].sum().reset_index()
            toyota_outflow.rename(columns={WEIGHT_LOYALTY: 'Value'}, inplace=True)
            
            for year in ['2018', '2025']:
                year_data = toyota_outflow[toyota_outflow['Year'] == year].nlargest(15, 'Value')
                
                fig_to = px.bar(year_data, x='Value', y='Brand_P', orientation='h',
                               title=f"Toyota 流出先 ({year}年)",
                               color_discrete_sequence=['#3498db' if year == '2018' else '#e74c3c'])
                fig_to.update_layout(height=500, yaxis={'categoryorder':'total ascending'},
                                    xaxis_title="ボリューム", yaxis_title="")
                st.plotly_chart(fig_to, use_container_width=True)
    
    with tab2:
        st.subheader("👤 顧客属性の変化")
        st.info(f"💡 顧客属性分析には **{WEIGHT_DISTRIBUTION}** を使用")
        
        demo_attr = st.selectbox("属性を選択", ["Income", "Age", "Lifestage"])
        
        # 年齢グループ化関数（Ageの場合に使用）
        def group_age(age_val):
            if pd.isna(age_val):
                return None
            age_str = str(age_val)
            
            # 文字列形式の処理
            if 'Under 20' in age_str or age_str in ['16.0', '17.0', '18.0', '19.0']:
                return '<20'
            elif '70 Or Over' in age_str:
                return '70+'
            elif ' To ' in age_str:
                # "20 To 24" のような形式を "20-24" に変換
                parts = age_str.split()
                if len(parts) >= 3:
                    return f'{parts[0]}-{parts[2]}'
            
            # 数値形式の処理
            try:
                age_num = float(age_str.replace('.0', ''))
                if age_num < 20:
                    return '<20'
                elif age_num >= 70:
                    return '70+'
                else:
                    lower = int(age_num // 5 * 5)
                    upper = lower + 4
                    return f'{lower}-{upper}'
            except:
                return None
        
        # Ageの場合は年齢グループ化を適用
        if demo_attr == "Age":
            # 年齢グループ化したデータを作成
            age_grouped_df = combined_df.copy()
            age_grouped_df['Age_Group'] = age_grouped_df['Age'].apply(group_age)
            age_grouped_df = age_grouped_df[age_grouped_df['Age_Group'].notna()]
            
            # 年齢グループの順序を定義
            age_order = ['<20', '20-24', '25-29', '30-34', '35-39', '40-44',
                        '45-49', '50-54', '55-59', '60-64', '65-69', '70+']
            
            # グループ化して集計
            demo_stats = get_weighted_share(age_grouped_df, ['Brand_P', 'Year'], 'Age_Group', WEIGHT_DISTRIBUTION)
            
            # 実際に使用する列名を設定（後続の処理で使用）
            actual_col = 'Age_Group'
            
            # ブランド別に並べて比較
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Honda", "Toyota"))
            
            for i, brand in enumerate(['Honda', 'Toyota'], 1):
                brand_data = demo_stats[demo_stats['Brand_P'] == brand]
                
                for year in ['2018', '2025']:
                    year_data = brand_data[brand_data['Year'] == year]
                    # 年齢グループの順序でソート
                    year_data['Age_Group'] = pd.Categorical(year_data['Age_Group'],
                                                            categories=age_order,
                                                            ordered=True)
                    year_data = year_data.sort_values('Age_Group')
                    
                    fig.add_trace(
                        go.Bar(x=year_data['Age_Group'], y=year_data['Share (%)'],
                               name=f"{year}", legendgroup=year, showlegend=(i==1)),
                        row=1, col=i
                    )
            
            fig.update_layout(height=500, barmode='group', title_text="年齢グループ 構成比較")
            fig.update_xaxes(categoryorder='array', categoryarray=age_order)
            st.plotly_chart(fig, use_container_width=True)
            
        elif demo_attr == "Income":
            # Incomeの場合は年収グループ化を適用
            income_grouped_df = combined_df.copy()
            income_grouped_df['Income_Group'] = income_grouped_df['Income'].apply(group_income_for_matrix)
            income_grouped_df = income_grouped_df[income_grouped_df['Income_Group'].notna()]
            
            # 年収グループの順序を定義（低い順）
            income_order = ['<$50k', '$50-99k', '$100-149k', '$150-199k',
                           '$200-299k', '$300-399k', '$400-499k', '$500k+']
            
            # グループ化して集計
            demo_stats = get_weighted_share(income_grouped_df, ['Brand_P', 'Year'], 'Income_Group', WEIGHT_DISTRIBUTION)
            
            # 実際に使用する列名を設定
            actual_col = 'Income_Group'
            
            # ブランド別に並べて比較
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Honda", "Toyota"))
            
            for i, brand in enumerate(['Honda', 'Toyota'], 1):
                brand_data = demo_stats[demo_stats['Brand_P'] == brand]
                
                for year in ['2018', '2025']:
                    year_data = brand_data[brand_data['Year'] == year]
                    # 年収グループの順序でソート
                    year_data['Income_Group'] = pd.Categorical(year_data['Income_Group'],
                                                               categories=income_order,
                                                               ordered=True)
                    year_data = year_data.sort_values('Income_Group')
                    
                    fig.add_trace(
                        go.Bar(x=year_data['Income_Group'], y=year_data['Share (%)'],
                               name=f"{year}", legendgroup=year, showlegend=(i==1)),
                        row=1, col=i
                    )
            
            fig.update_layout(height=500, barmode='group', title_text="年収グループ 構成比較")
            fig.update_xaxes(categoryorder='array', categoryarray=income_order)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Lifestageなど他の属性は既存の処理
            demo_stats = get_weighted_share(combined_df, ['Brand_P', 'Year'], demo_attr, WEIGHT_DISTRIBUTION)
            
            # 実際に使用する列名を設定
            actual_col = demo_attr
            
            # ブランド別に並べて比較
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Honda", "Toyota"))
            
            for i, brand in enumerate(['Honda', 'Toyota'], 1):
                brand_data = demo_stats[demo_stats['Brand_P'] == brand]
                
                for year in ['2018', '2025']:
                    year_data = brand_data[brand_data['Year'] == year]
                    fig.add_trace(
                        go.Bar(x=year_data[demo_attr], y=year_data['Share (%)'],
                               name=f"{year}", legendgroup=year, showlegend=(i==1)),
                        row=1, col=i
                    )
            
            fig.update_layout(height=500, barmode='group', title_text=f"{demo_attr} 構成比較")
            st.plotly_chart(fig, use_container_width=True)
        
        # Incomeが選択された場合のみ、年齢×年収マトリクスを追加表示
        if demo_attr == "Income":
            st.divider()
            st.markdown("### 📊 年齢×年収マトリクス")
            st.caption("Honda と Toyota の顧客の年齢層と年収層の分布を割合(%)で可視化")
            
            # データ準備
            matrix_df = combined_df[combined_df['Age'].notna() & combined_df['Income'].notna()].copy()
            
            # グループ化を適用
            matrix_df['Age_Group'] = matrix_df['Age'].apply(group_age)
            matrix_df['Income_Group'] = matrix_df['Income'].apply(group_income_for_matrix)
            
            # 年齢と年収でクロス集計（軸を入れ替え: 横軸=年齢、縦軸=年収）
            def create_age_income_matrix(data, year_val):
                year_data = data[data['Year'] == year_val]
                year_data = year_data[year_data['Age_Group'].notna() & year_data['Income_Group'].notna()]
                
                pivot = year_data.pivot_table(
                    values=WEIGHT_DISTRIBUTION,
                    index='Income_Group',  # 縦軸: 年収
                    columns='Age_Group',   # 横軸: 年齢
                    aggfunc='sum',
                    fill_value=0
                )
                
                # 常に割合に変換
                total = pivot.sum().sum()
                if total > 0:
                    pivot = (pivot / total) * 100
                
                # 年齢と年収の順序を定義
                age_order = ['<20', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                            '50-54', '55-59', '60-64', '65-69', '70+']
                # 年収は上が高く、下が低くなるように逆順
                income_order = ['$500k+', '$400-499k', '$300-399k', '$200-299k',
                               '$150-199k', '$100-149k', '$50-99k', '<$50k']
                
                # 存在する列・行のみを使用して並び替え
                existing_ages = [a for a in age_order if a in pivot.columns]
                existing_incomes = [i for i in income_order if i in pivot.index]
                
                pivot = pivot.reindex(index=existing_incomes, columns=existing_ages, fill_value=0)
                
                return pivot
            
            # Honda/Toyota別に表示
            col_h, col_t = st.columns(2)
            
            with col_h:
                st.markdown("#### 🔵 Honda")
                honda_matrix = matrix_df[matrix_df['Brand_P'] == 'Honda']
                
                if len(honda_matrix) > 0:
                    # 2018年
                    matrix_2018 = create_age_income_matrix(honda_matrix, '2018')
                    # 2025年
                    matrix_2025 = create_age_income_matrix(honda_matrix, '2025')
                    
                    # 年齢と年収の軸を統一（両年のユニオンを取る）
                    all_ages = matrix_2018.columns.union(matrix_2025.columns)
                    all_incomes = matrix_2018.index.union(matrix_2025.index)
                    matrix_2018 = matrix_2018.reindex(index=all_incomes, columns=all_ages, fill_value=0)
                    matrix_2025 = matrix_2025.reindex(index=all_incomes, columns=all_ages, fill_value=0)
                    
                    # 差分計算
                    matrix_diff = matrix_2025 - matrix_2018
                    
                    # 3つのヒートマップを縦に並べる
                    st.markdown("**2018年**")
                    vmin_2018 = matrix_2018.values[matrix_2018.values > 0].min() if (matrix_2018.values > 0).any() else 0
                    vmax_2018 = np.percentile(matrix_2018.values[matrix_2018.values > 0], 95) if (matrix_2018.values > 0).any() else matrix_2018.values.max()
                    
                    fig_2018 = px.imshow(matrix_2018,
                                        labels=dict(x="年齢", y="年収", color="割合 (%)"),
                                        aspect="auto",
                                        color_continuous_scale="RdBu_r",
                                        zmin=vmin_2018,
                                        zmax=vmax_2018,
                                        text_auto='.1f')
                    fig_2018.update_layout(height=400)
                    st.plotly_chart(fig_2018, use_container_width=True)
                    
                    st.markdown("**2025年**")
                    vmin_2025 = matrix_2025.values[matrix_2025.values > 0].min() if (matrix_2025.values > 0).any() else 0
                    vmax_2025 = np.percentile(matrix_2025.values[matrix_2025.values > 0], 95) if (matrix_2025.values > 0).any() else matrix_2025.values.max()
                    
                    fig_2025 = px.imshow(matrix_2025,
                                        labels=dict(x="年齢", y="年収", color="割合 (%)"),
                                        aspect="auto",
                                        color_continuous_scale="RdBu_r",
                                        zmin=vmin_2025,
                                        zmax=vmax_2025,
                                        text_auto='.1f')
                    fig_2025.update_layout(height=400)
                    st.plotly_chart(fig_2025, use_container_width=True)
                    
                    st.markdown("**差分 (2025 - 2018)**")
                    abs_max_diff = np.percentile(np.abs(matrix_diff.values), 95)
                    
                    fig_diff = px.imshow(matrix_diff,
                                        labels=dict(x="年齢", y="年収", color="変化量 (%)"),
                                        aspect="auto",
                                        color_continuous_scale="RdBu_r",
                                        zmin=-abs_max_diff,
                                        zmax=abs_max_diff,
                                        color_continuous_midpoint=0,
                                        text_auto='.1f')
                    fig_diff.update_layout(height=400)
                    st.plotly_chart(fig_diff, use_container_width=True)
            
            with col_t:
                st.markdown("#### 🔴 Toyota")
                toyota_matrix = matrix_df[matrix_df['Brand_P'] == 'Toyota']
                
                if len(toyota_matrix) > 0:
                    # 2018年
                    matrix_2018 = create_age_income_matrix(toyota_matrix, '2018')
                    # 2025年
                    matrix_2025 = create_age_income_matrix(toyota_matrix, '2025')
                    
                    # 年齢と年収の軸を統一（両年のユニオンを取る）
                    all_ages = matrix_2018.columns.union(matrix_2025.columns)
                    all_incomes = matrix_2018.index.union(matrix_2025.index)
                    matrix_2018 = matrix_2018.reindex(index=all_incomes, columns=all_ages, fill_value=0)
                    matrix_2025 = matrix_2025.reindex(index=all_incomes, columns=all_ages, fill_value=0)
                    
                    # 差分計算
                    matrix_diff = matrix_2025 - matrix_2018
                    
                    # 3つのヒートマップを縦に並べる
                    st.markdown("**2018年**")
                    vmin_2018 = matrix_2018.values[matrix_2018.values > 0].min() if (matrix_2018.values > 0).any() else 0
                    vmax_2018 = np.percentile(matrix_2018.values[matrix_2018.values > 0], 95) if (matrix_2018.values > 0).any() else matrix_2018.values.max()
                    
                    fig_2018 = px.imshow(matrix_2018,
                                        labels=dict(x="年齢", y="年収", color="割合 (%)"),
                                        aspect="auto",
                                        color_continuous_scale="RdBu_r",
                                        zmin=vmin_2018,
                                        zmax=vmax_2018,
                                        text_auto='.1f')
                    fig_2018.update_layout(height=400)
                    st.plotly_chart(fig_2018, use_container_width=True)
                    
                    st.markdown("**2025年**")
                    vmin_2025 = matrix_2025.values[matrix_2025.values > 0].min() if (matrix_2025.values > 0).any() else 0
                    vmax_2025 = np.percentile(matrix_2025.values[matrix_2025.values > 0], 95) if (matrix_2025.values > 0).any() else matrix_2025.values.max()
                    
                    fig_2025 = px.imshow(matrix_2025,
                                        labels=dict(x="年齢", y="年収", color="割合 (%)"),
                                        aspect="auto",
                                        color_continuous_scale="RdBu_r",
                                        zmin=vmin_2025,
                                        zmax=vmax_2025,
                                        text_auto='.1f')
                    fig_2025.update_layout(height=400)
                    st.plotly_chart(fig_2025, use_container_width=True)
                    
                    st.markdown("**差分 (2025年間推定 - 2018)**")
                    st.caption("※2025年はW3まで（9ヶ月）のデータを年間換算")
                    abs_max_diff = np.percentile(np.abs(matrix_diff.values), 95)
                    
                    fig_diff = px.imshow(matrix_diff,
                                        labels=dict(x="年齢", y="年収", color="変化量 (%)"),
                                        aspect="auto",
                                        color_continuous_scale="RdBu_r",
                                        zmin=-abs_max_diff,
                                        zmax=abs_max_diff,
                                        color_continuous_midpoint=0,
                                        text_auto='.1f')
                    fig_diff.update_layout(height=400)
                    st.plotly_chart(fig_diff, use_container_width=True)
        
        # 統計サマリー
        st.markdown("### 📊 変化のハイライト")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Honda**")
            honda_data = demo_stats[demo_stats['Brand_P'] == 'Honda']
            top_2018 = honda_data[honda_data['Year'] == '2018'].nlargest(3, 'Share (%)')[actual_col].tolist()
            top_2025 = honda_data[honda_data['Year'] == '2025'].nlargest(3, 'Share (%)')[actual_col].tolist()
            st.write(f"2018 TOP3: {', '.join(map(str, top_2018))}")
            st.write(f"2025 TOP3: {', '.join(map(str, top_2025))}")
        
        with col2:
            st.markdown("**Toyota**")
            toyota_data = demo_stats[demo_stats['Brand_P'] == 'Toyota']
            top_2018 = toyota_data[toyota_data['Year'] == '2018'].nlargest(3, 'Share (%)')[actual_col].tolist()
            top_2025 = toyota_data[toyota_data['Year'] == '2025'].nlargest(3, 'Share (%)')[actual_col].tolist()
            st.write(f"2018 TOP3: {', '.join(map(str, top_2018))}")
            st.write(f"2025 TOP3: {', '.join(map(str, top_2025))}")
    
    with tab3:
        st.subheader("🗺 地域分布の変化")
        st.info(f"💡 地域分布分析には **{WEIGHT_DISTRIBUTION}** を使用")
        
        # HondaとToyotaのみにフィルタリング
        geo_df = combined_df[combined_df['Brand_P'].isin(['Honda', 'Toyota'])].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 居住地特性 (Urbanicity)")
            urb_stats = get_weighted_share(geo_df, ['Brand_P', 'Year'], 'Urbanicity')
            fig_urb = px.bar(urb_stats, x='Urbanicity', y='Share (%)', color='Brand_P',
                            barmode='group', facet_col='Year', title="居住地特性の比較")
            st.plotly_chart(fig_urb, use_container_width=True)
        
        with col2:
            st.markdown("### リージョン別分布")
            reg_stats = get_weighted_share(geo_df, ['Brand_P', 'Year'], 'Region')
            fig_reg = px.bar(reg_stats, x='Region', y='Share (%)', color='Brand_P',
                            barmode='group', facet_col='Year', title="リージョン別分布")
            st.plotly_chart(fig_reg, use_container_width=True)
    
    with tab4:
        st.subheader("💰 支払方法の変化")
        st.info(f"💡 支払方法分析には **{WEIGHT_DISTRIBUTION}** を使用")
        
        # HondaとToyotaのみにフィルタリング
        payment_df = combined_df[combined_df['Brand_P'].isin(['Honda', 'Toyota'])].copy()
        
        st.markdown("### 支払方法")
        pay_stats = get_weighted_share(payment_df, ['Brand_P', 'Year'], 'Payment_Method')
        fig_pay = px.bar(pay_stats, x='Payment_Method', y='Share (%)', color='Brand_P',
                        barmode='group', facet_col='Year', title="支払方法の比較")
        st.plotly_chart(fig_pay, use_container_width=True)
    
    # ==================== タブ5: 離反理由 ====================
    with tab5:
        st.subheader("❓ 離反理由の分析")
        st.info(f"💡 離反理由分析には **{WEIGHT_LOYALTY}** を使用")
        
        # Detailed_Statusを使って離反者を抽出（combined_dfから）
        defectors = combined_df[combined_df['Detailed_Status'].str.contains('Defection', case=False, na=False)].copy()
        
        if len(defectors) > 0:
            # Honda離反者とToyota離反者を分離
            honda_defectors = defectors[defectors['Detailed_Status'].str.contains('from Honda', case=False, na=False)]
            toyota_defectors = defectors[defectors['Detailed_Status'].str.contains('from Toyota', case=False, na=False)]
            
            st.markdown(f"**離反者数**: Honda {len(honda_defectors):,}名 / Toyota {len(toyota_defectors):,}名")
            
            # 2列レイアウト: Honda / Toyota
            col_h, col_t = st.columns(2)
            
            # 新しいWhy NOT Shop列名を取得
            why_not_shop_cols = [col for col in defectors.columns if 'Why NOT Shop' in col]
            
            with col_h:
                st.markdown("### 🔵 Honda 離反理由")
                
                if len(honda_defectors) > 0 and len(why_not_shop_cols) > 0:
                    # Why NOT Shop列を集約
                    reasons_list = []
                    for col_name in why_not_shop_cols:
                        if col_name in honda_defectors.columns:
                            temp = honda_defectors.groupby(['Year', col_name])[WEIGHT_LOYALTY].sum().reset_index()
                            temp.rename(columns={col_name: 'Reason', WEIGHT_LOYALTY: 'Value'}, inplace=True)
                            reasons_list.append(temp)
                    
                    if reasons_list:
                        honda_reasons = pd.concat(reasons_list, ignore_index=True)
                        honda_reasons = honda_reasons[honda_reasons['Reason'].notna()]
                        honda_reasons = honda_reasons.groupby(['Year', 'Reason'])['Value'].sum().reset_index()
                        
                        # 2018年と2025年で分けて表示
                        for year in [2018, 2025]:
                            year_data = honda_reasons[honda_reasons['Year'] == year].copy()
                            if len(year_data) > 0:
                                year_data = year_data.sort_values('Value', ascending=False).head(10)
                                
                                st.markdown(f"**{year}年 TOP10**")
                                fig_h = px.bar(year_data,
                                              y='Reason',
                                              x='Value',
                                              orientation='h',
                                              title=f"Honda離反理由 ({year})",
                                              color_discrete_sequence=['#3498db'])
                                fig_h.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                                st.plotly_chart(fig_h, use_container_width=True)
                    else:
                        st.info("離反理由データがありません")
                else:
                    st.info("Honda離反者データがありません")
            
            with col_t:
                st.markdown("### 🔴 Toyota 離反理由")
                
                if len(toyota_defectors) > 0 and len(why_not_shop_cols) > 0:
                    # Why NOT Shop列を集約
                    reasons_list = []
                    for col_name in why_not_shop_cols:
                        if col_name in toyota_defectors.columns:
                            temp = toyota_defectors.groupby(['Year', col_name])[WEIGHT_LOYALTY].sum().reset_index()
                            temp.rename(columns={col_name: 'Reason', WEIGHT_LOYALTY: 'Value'}, inplace=True)
                            reasons_list.append(temp)
                    
                    if reasons_list:
                        toyota_reasons = pd.concat(reasons_list, ignore_index=True)
                        toyota_reasons = toyota_reasons[toyota_reasons['Reason'].notna()]
                        toyota_reasons = toyota_reasons.groupby(['Year', 'Reason'])['Value'].sum().reset_index()
                        
                        # 2018年と2025年で分けて表示
                        for year in [2018, 2025]:
                            year_data = toyota_reasons[toyota_reasons['Year'] == year].copy()
                            if len(year_data) > 0:
                                year_data = year_data.sort_values('Value', ascending=False).head(10)
                                
                                st.markdown(f"**{year}年 TOP10**")
                                fig_t = px.bar(year_data,
                                              y='Reason',
                                              x='Value',
                                              orientation='h',
                                              title=f"Toyota離反理由 ({year})",
                                              color_discrete_sequence=['#e74c3c'])
                                fig_t.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                                st.plotly_chart(fig_t, use_container_width=True)
                    else:
                        st.info("離反理由データがありません")
                else:
                    st.info("Toyota離反者データがありません")
        else:
            st.warning("離反者データが見つかりません")

# ==================== モード2: モデル別分析 ====================
elif analysis_mode == "🚗 モデル別分析":
    st.title("🚗 モデル別詳細分析 (2018 vs 2025)")
    
    # ブランド選択
    brand = st.sidebar.radio("ブランド選択", ["Honda", "Toyota"])
    
    # モデルリスト（部分一致で検索）
    major_models = HONDA_MAJOR if brand == "Honda" else TOYOTA_MAJOR
    brand_df = df[df['Brand_P'] == brand].copy()
    brand_df = brand_df[brand_df['Model_P'].str.contains('|'.join(major_models), case=False, na=False)]
    
    # モデル名を簡略化（グレード情報を除去）
    def simplify_model_name(full_name):
        if pd.isna(full_name):
            return None
        for model in major_models:
            if model in full_name:
                return model
        return full_name
    
    brand_df['Model_Simple'] = brand_df['Model_P'].apply(simplify_model_name)
    
    # モデル選択
    available_models = sorted(brand_df['Model_Simple'].dropna().unique())
    selected_model = st.sidebar.selectbox("分析対象モデル", available_models)
    model_df = brand_df[brand_df['Model_Simple'] == selected_model].copy()
    
    st.markdown(f"## {brand} {selected_model} の変化")
    
    # タブ構成
    tab1, tab2, tab3, tab4 = st.tabs(["📊 ロイヤリティ・流入流出", "👤 顧客属性", "🗺 地域", "💰 価格"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔄 離反率")
            st.caption(f"💡 {WEIGHT_LOYALTY} を使用")
            # 離反率 = このモデルを処分した人のうち、他ブランドに流出した人の割合
            # Model_D（処分）がselected_modelの人を分母とする
            defection_data = []
            for year in ['2018', '2025']:
                # このモデルを処分した人全体
                disposed_df = df[df['Year'] == year].copy()
                disposed_df['Model_D_Simple'] = disposed_df['Model_D'].apply(simplify_model_name)
                disposed_model = disposed_df[disposed_df['Model_D_Simple'] == selected_model]
                
                if len(disposed_model) > 0:
                    total_weight = disposed_model[WEIGHT_LOYALTY].sum()
                    # 離反 = 処分したモデルと購入したブランドが異なる
                    defection_weight = disposed_model[disposed_model['Detailed_Status'].str.contains('Defection', case=False, na=False)][WEIGHT_LOYALTY].sum()
                    defection_rate = (defection_weight / total_weight * 100) if total_weight > 0 else 0
                    defection_data.append({'Year': year, 'Share (%)': defection_rate, 'Total': int(total_weight)})
                else:
                    defection_data.append({'Year': year, 'Share (%)': 0, 'Total': 0})
            
            defection_stats = pd.DataFrame(defection_data)
            
            fig_def = px.bar(defection_stats, x='Year', y='Share (%)', text='Share (%)',
                            color='Year', color_discrete_map={'2018': '#95a5a6', '2025': '#e74c3c'})
            fig_def.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            fig_def.update_layout(title=f"離反率 (分母: このモデルを処分した人)")
            st.plotly_chart(fig_def, use_container_width=True)
            
            # 参考情報を表示
            st.caption(f"2018年: {defection_stats[defection_stats['Year']=='2018']['Total'].values[0]:,.0f}人が処分")
            st.caption(f"2025年: {defection_stats[defection_stats['Year']=='2025']['Total'].values[0]:,.0f}人が処分")
        
        with col2:
            st.markdown("### 📥 流入元モデル TOP20")
            st.caption(f"💡 {WEIGHT_LOYALTY} を使用")
            
            # フィルタ選択
            brand_filter = st.radio(
                "表示するモデル",
                ["すべて", "自社のみ", "他社のみ"],
                horizontal=True,
                key='inflow_filter'
            )
            
            # モデル名簡略化関数（グレード除去）
            def simplify_model_for_aggregation(model_name):
                if pd.isna(model_name):
                    return None
                model_str = str(model_name)
                # Honda/Acuraモデル
                for honda_model in HONDA_MAJOR:
                    if honda_model.upper() in model_str.upper():
                        return honda_model
                # Toyotaモデル
                for toyota_model in TOYOTA_MAJOR:
                    if toyota_model.upper() in model_str.upper():
                        return toyota_model
                # その他はそのまま返す
                return model_str
            
            # ブランド判定関数
            def get_brand_from_model(model_name):
                if pd.isna(model_name):
                    return None
                model_str = str(model_name).upper()
                for honda_model in HONDA_MAJOR:
                    if honda_model.upper() in model_str:
                        return 'Honda'
                if 'ACURA' in model_str:
                    return 'Honda'
                for toyota_model in TOYOTA_MAJOR:
                    if toyota_model.upper() in model_str:
                        return 'Toyota'
                return 'Other'
            
            for year in ['2018', '2025']:
                year_data = model_df[model_df['Year'] == year].copy()
                # 流入元：このモデルを購入した人が以前乗っていたモデル
                year_data['Model_D_Simple'] = year_data['Model_D'].apply(simplify_model_for_aggregation)
                year_data['ブランド'] = year_data['Model_D_Simple'].apply(get_brand_from_model)
                
                # フィルタ適用
                if brand_filter == "自社のみ":
                    year_data = year_data[year_data['ブランド'] == brand]
                elif brand_filter == "他社のみ":
                    year_data = year_data[year_data['ブランド'] != brand]
                
                inflow = year_data[year_data['Model_D_Simple'].notna()].groupby('Model_D_Simple')[WEIGHT_LOYALTY].sum().nlargest(20).reset_index()
                st.markdown(f"**{year}年**")
                if len(inflow) > 0:
                    # ブランド判定を追加
                    inflow['ブランド'] = inflow['Model_D_Simple'].apply(get_brand_from_model)
                    # ボリュームを四捨五入
                    inflow[WEIGHT_LOYALTY] = inflow[WEIGHT_LOYALTY].round(0).astype(int)
                    inflow_display = inflow.rename(columns={'Model_D_Simple': 'モデル', WEIGHT_LOYALTY: 'ボリューム'})
                    
                    # 同一ブランドの行をグレーでハイライト
                    def highlight_same_brand(row):
                        if row['ブランド'] == brand:
                            return ['background-color: #f0f0f0'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(inflow_display.style.apply(highlight_same_brand, axis=1),
                               hide_index=True, height=400)
                else:
                    st.info("データなし")
        
        st.markdown("### 📤 流出先モデル TOP20")
        st.caption(f"💡 {WEIGHT_LOYALTY} を使用")
        
        # フィルタ選択
        brand_filter_out = st.radio(
            "表示するモデル",
            ["すべて", "自社のみ", "他社のみ"],
            horizontal=True,
            key='outflow_filter'
        )
        
        col3, col4 = st.columns(2)
        
        # 流出先は、このモデルを処分した人が次に買ったモデル
        # Model_D（処分）がselected_modelで、Model_P（購入）が流出先
        outflow_df = df[df['Model_D'].notna()].copy()
        # モデル名を簡略化
        outflow_df['Model_D_Simple'] = outflow_df['Model_D'].apply(simplify_model_name)
        outflow_df = outflow_df[outflow_df['Model_D_Simple'] == selected_model]
        
        # モデル名簡略化関数（グレード除去）
        def simplify_model_for_aggregation(model_name):
            if pd.isna(model_name):
                return None
            model_str = str(model_name)
            # Honda/Acuraモデル
            for honda_model in HONDA_MAJOR:
                if honda_model.upper() in model_str.upper():
                    return honda_model
            # Toyotaモデル
            for toyota_model in TOYOTA_MAJOR:
                if toyota_model.upper() in model_str.upper():
                    return toyota_model
            # その他はそのまま返す
            return model_str
        
        # ブランド判定関数
        def get_brand_from_model(model_name):
            if pd.isna(model_name):
                return None
            model_str = str(model_name).upper()
            for honda_model in HONDA_MAJOR:
                if honda_model.upper() in model_str:
                    return 'Honda'
            if 'ACURA' in model_str:
                return 'Honda'
            for toyota_model in TOYOTA_MAJOR:
                if toyota_model.upper() in model_str:
                    return 'Toyota'
            return 'Other'
        
        with col3:
            st.markdown("**2018年**")
            out_2018 = outflow_df[outflow_df['Year'] == '2018'].copy()
            if len(out_2018) > 0:
                # モデル名を簡略化してから集計
                out_2018['Model_P_Simple'] = out_2018['Model_P'].apply(simplify_model_for_aggregation)
                out_2018['ブランド'] = out_2018['Model_P_Simple'].apply(get_brand_from_model)
                
                # フィルタ適用
                if brand_filter_out == "自社のみ":
                    out_2018 = out_2018[out_2018['ブランド'] == brand]
                elif brand_filter_out == "他社のみ":
                    out_2018 = out_2018[out_2018['ブランド'] != brand]
                
                out_2018_top = out_2018.groupby('Model_P_Simple')[WEIGHT_LOYALTY].sum().nlargest(20).reset_index()
                out_2018_top['ブランド'] = out_2018_top['Model_P_Simple'].apply(get_brand_from_model)
                # ボリュームを四捨五入
                out_2018_top[WEIGHT_LOYALTY] = out_2018_top[WEIGHT_LOYALTY].round(0).astype(int)
                out_2018_display = out_2018_top.rename(columns={'Model_P_Simple': 'モデル', WEIGHT_LOYALTY: 'ボリューム'})
                
                # 同一ブランドの行をグレーでハイライト
                def highlight_same_brand(row):
                    if row['ブランド'] == brand:
                        return ['background-color: #f0f0f0'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(out_2018_display.style.apply(highlight_same_brand, axis=1),
                            hide_index=True, height=400)
            else:
                st.info("データなし")
        
        with col4:
            st.markdown("**2025年**")
            out_2025 = outflow_df[outflow_df['Year'] == '2025'].copy()
            if len(out_2025) > 0:
                # モデル名を簡略化してから集計
                out_2025['Model_P_Simple'] = out_2025['Model_P'].apply(simplify_model_for_aggregation)
                out_2025['ブランド'] = out_2025['Model_P_Simple'].apply(get_brand_from_model)
                
                # フィルタ適用
                if brand_filter_out == "自社のみ":
                    out_2025 = out_2025[out_2025['ブランド'] == brand]
                elif brand_filter_out == "他社のみ":
                    out_2025 = out_2025[out_2025['ブランド'] != brand]
                
                out_2025_top = out_2025.groupby('Model_P_Simple')[WEIGHT_LOYALTY].sum().nlargest(20).reset_index()
                out_2025_top['ブランド'] = out_2025_top['Model_P_Simple'].apply(get_brand_from_model)
                # ボリュームを四捨五入
                out_2025_top[WEIGHT_LOYALTY] = out_2025_top[WEIGHT_LOYALTY].round(0).astype(int)
                out_2025_display = out_2025_top.rename(columns={'Model_P_Simple': 'モデル', WEIGHT_LOYALTY: 'ボリューム'})
                
                # 同一ブランドの行をグレーでハイライト
                def highlight_same_brand(row):
                    if row['ブランド'] == brand:
                        return ['background-color: #f0f0f0'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(out_2025_display.style.apply(highlight_same_brand, axis=1),
                            hide_index=True, height=400)
            else:
                st.info("データなし")
    
    with tab2:
        st.markdown("### 👤 顧客属性の変化")
        
        demo_attr = st.selectbox("属性", ["Income", "Age", "Lifestage"], key='model_demo')
        
        # Ageの場合は年齢グループ化を適用
        if demo_attr == "Age":
            # 年齢グループ化関数（ブランド全体比較と同じ）
            def group_age_model(age_val):
                if pd.isna(age_val):
                    return None
                age_str = str(age_val)
                
                if 'Under 20' in age_str or age_str in ['16.0', '17.0', '18.0', '19.0']:
                    return '<20'
                elif '70 Or Over' in age_str:
                    return '70+'
                elif ' To ' in age_str:
                    parts = age_str.split()
                    if len(parts) >= 3:
                        return f'{parts[0]}-{parts[2]}'
                
                try:
                    age_num = float(age_str.replace('.0', ''))
                    if age_num < 20:
                        return '<20'
                    elif age_num >= 70:
                        return '70+'
                    else:
                        lower = int(age_num // 5 * 5)
                        upper = lower + 4
                        return f'{lower}-{upper}'
                except:
                    return None
            
            # 年齢グループ化したデータを作成
            age_grouped_df = model_df.copy()
            age_grouped_df['Age_Group'] = age_grouped_df['Age'].apply(group_age_model)
            age_grouped_df = age_grouped_df[age_grouped_df['Age_Group'].notna()]
            
            # 年齢グループの順序を定義
            age_order = ['<20', '20-24', '25-29', '30-34', '35-39', '40-44',
                        '45-49', '50-54', '55-59', '60-64', '65-69', '70+']
            
            # グループ化して集計
            demo_stats = get_weighted_share(age_grouped_df, 'Year', 'Age_Group')
            
            # 年齢グループの順序でソート
            demo_stats['Age_Group'] = pd.Categorical(demo_stats['Age_Group'],
                                                     categories=age_order,
                                                     ordered=True)
            demo_stats = demo_stats.sort_values('Age_Group')
            
            fig_demo = px.bar(demo_stats, x='Age_Group', y='Share (%)', color='Year',
                             barmode='group', title=f"{selected_model} の年齢グループ構成",
                             category_orders={'Age_Group': age_order})
            st.plotly_chart(fig_demo, use_container_width=True)
        elif demo_attr == "Income":
            # Incomeの場合は年収グループ化を適用
            income_grouped_df = model_df.copy()
            income_grouped_df['Income_Group'] = income_grouped_df['Income'].apply(group_income_for_matrix)
            income_grouped_df = income_grouped_df[income_grouped_df['Income_Group'].notna()]
            
            # 年収グループの順序を定義（低い順）
            income_order = ['<$50k', '$50-99k', '$100-149k', '$150-199k',
                           '$200-299k', '$300-399k', '$400-499k', '$500k+']
            
            # グループ化して集計
            demo_stats = get_weighted_share(income_grouped_df, 'Year', 'Income_Group')
            
            # 年収グループの順序でソート
            demo_stats['Income_Group'] = pd.Categorical(demo_stats['Income_Group'],
                                                        categories=income_order,
                                                        ordered=True)
            demo_stats = demo_stats.sort_values('Income_Group')
            
            fig_demo = px.bar(demo_stats, x='Income_Group', y='Share (%)', color='Year',
                             barmode='group', title=f"{selected_model} の年収グループ構成",
                             category_orders={'Income_Group': income_order})
            st.plotly_chart(fig_demo, use_container_width=True)
        else:
            # Lifestageなど他の属性は既存の処理
            demo_stats = get_weighted_share(model_df, 'Year', demo_attr)
            
            fig_demo = px.bar(demo_stats, x=demo_attr, y='Share (%)', color='Year',
                             barmode='group', title=f"{selected_model} の {demo_attr} 構成")
            st.plotly_chart(fig_demo, use_container_width=True)
        
        # Incomeが選択された場合のみ、年齢×年収マトリクスを追加表示
        if demo_attr == "Income":
            st.divider()
            st.markdown("### 📊 年齢×年収マトリクス")
            st.caption(f"{selected_model} の顧客の年齢層と年収層の分布を割合(%)で可視化")
            
            # データ準備
            matrix_df_model = model_df[model_df['Age'].notna() & model_df['Income'].notna()].copy()
            
            if len(matrix_df_model) > 0:
                # 年齢グループ化関数
                def group_age_matrix(age_val):
                    if pd.isna(age_val):
                        return None
                    age_str = str(age_val)
                    if 'Under 20' in age_str or age_str in ['16.0', '17.0', '18.0', '19.0']:
                        return '<20'
                    elif '70 Or Over' in age_str:
                        return '70+'
                    elif ' To ' in age_str:
                        parts = age_str.split()
                        if len(parts) >= 3:
                            return f'{parts[0]}-{parts[2]}'
                    try:
                        age_num = float(age_str.replace('.0', ''))
                        if age_num < 20:
                            return '<20'
                        elif age_num >= 70:
                            return '70+'
                        else:
                            lower = int(age_num // 5 * 5)
                            upper = lower + 4
                            return f'{lower}-{upper}'
                    except:
                        return None
                
                # グループ化を適用
                matrix_df_model['Age_Group'] = matrix_df_model['Age'].apply(group_age_matrix)
                matrix_df_model['Income_Group'] = matrix_df_model['Income'].apply(group_income_for_matrix)
                
                # マトリクス作成関数
                def create_model_matrix(data, year_val):
                    year_data = data[data['Year'] == year_val]
                    year_data = year_data[year_data['Age_Group'].notna() & year_data['Income_Group'].notna()]
                    
                    pivot = year_data.pivot_table(
                        values=WEIGHT_DISTRIBUTION,
                        index='Income_Group',
                        columns='Age_Group',
                        aggfunc='sum',
                        fill_value=0
                    )
                    
                    # 常に割合に変換
                    total = pivot.sum().sum()
                    if total > 0:
                        pivot = (pivot / total) * 100
                    
                    # 順序定義
                    age_order = ['<20', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                                '50-54', '55-59', '60-64', '65-69', '70+']
                    income_order = ['$500k+', '$400-499k', '$300-399k', '$200-299k',
                                   '$150-199k', '$100-149k', '$50-99k', '<$50k']
                    
                    existing_ages = [a for a in age_order if a in pivot.columns]
                    existing_incomes = [i for i in income_order if i in pivot.index]
                    
                    pivot = pivot.reindex(index=existing_incomes, columns=existing_ages, fill_value=0)
                    return pivot
                
                # 2018年と2025年のマトリクスを作成
                matrix_2018 = create_model_matrix(matrix_df_model, '2018')
                matrix_2025 = create_model_matrix(matrix_df_model, '2025')
                
                # 軸を統一
                all_ages = matrix_2018.columns.union(matrix_2025.columns)
                all_incomes = matrix_2018.index.union(matrix_2025.index)
                matrix_2018 = matrix_2018.reindex(index=all_incomes, columns=all_ages, fill_value=0)
                matrix_2025 = matrix_2025.reindex(index=all_incomes, columns=all_ages, fill_value=0)
                
                # 差分計算
                matrix_diff = matrix_2025 - matrix_2018
                
                # 3つのヒートマップを表示
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.markdown("**2018年**")
                    vmin = matrix_2018.values[matrix_2018.values > 0].min() if (matrix_2018.values > 0).any() else 0
                    vmax = np.percentile(matrix_2018.values[matrix_2018.values > 0], 95) if (matrix_2018.values > 0).any() else matrix_2018.values.max()
                    fig = px.imshow(matrix_2018, labels=dict(x="年齢", y="年収", color="割合 (%)"),
                                   aspect="auto", color_continuous_scale="RdBu_r",
                                   zmin=vmin, zmax=vmax, text_auto='.1f')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_m2:
                    st.markdown("**2025年**")
                    vmin = matrix_2025.values[matrix_2025.values > 0].min() if (matrix_2025.values > 0).any() else 0
                    vmax = np.percentile(matrix_2025.values[matrix_2025.values > 0], 95) if (matrix_2025.values > 0).any() else matrix_2025.values.max()
                    fig = px.imshow(matrix_2025, labels=dict(x="年齢", y="年収", color="割合 (%)"),
                                   aspect="auto", color_continuous_scale="RdBu_r",
                                   zmin=vmin, zmax=vmax, text_auto='.1f')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_m3:
                    st.markdown("**差分 (2025 - 2018)**")
                    abs_max = np.percentile(np.abs(matrix_diff.values), 95)
                    fig = px.imshow(matrix_diff, labels=dict(x="年齢", y="年収", color="変化量 (%)"),
                                   aspect="auto", color_continuous_scale="RdBu_r",
                                   zmin=-abs_max, zmax=abs_max, color_continuous_midpoint=0, text_auto='.1f')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("年齢×年収データが不足しています")
    
    with tab3:
        st.markdown("### 🗺 地域分布")
        
        col1, col2 = st.columns(2)
        
        with col1:
            urb_stats = get_weighted_share(model_df, 'Year', 'Urbanicity')
            fig_urb = px.bar(urb_stats, x='Urbanicity', y='Share (%)', color='Year',
                            barmode='group', title="居住地特性")
            st.plotly_chart(fig_urb, use_container_width=True)
        
        with col2:
            reg_stats = get_weighted_share(model_df, 'Year', 'Region')
            fig_reg = px.bar(reg_stats, x='Region', y='Share (%)', color='Year',
                            barmode='group', title="リージョン分布")
            st.plotly_chart(fig_reg, use_container_width=True)
    
    with tab4:
        st.markdown("### 💰 価格・支払")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pay_stats = get_weighted_share(model_df, 'Year', 'Payment_Method')
            fig_pay = px.bar(pay_stats, x='Payment_Method', y='Share (%)', color='Year',
                            barmode='group', title="支払方法")
            st.plotly_chart(fig_pay, use_container_width=True)
        
        with col2:
            # 平均価格（NaNと0を除外）
            price_data = []
            for year, group in model_df.groupby('Year'):
                try:
                    # Price列がNaNでなく、かつ0より大きいレコードのみを使用
                    valid_price = group[(group['Price'].notna()) & (group['Price'] > 0)].copy()
                    if len(valid_price) > 0:
                        avg_price = (valid_price['Price'].astype(float) * valid_price[WEIGHT_DISTRIBUTION]).sum() / valid_price[WEIGHT_DISTRIBUTION].sum()
                        price_data.append({'Year': year, 'Avg_Price': avg_price})
                except:
                    pass
            
            if len(price_data) > 0:
                price_avg = pd.DataFrame(price_data)
                
                fig_price = px.bar(price_avg, x='Year', y='Avg_Price', text='Avg_Price',
                                  color='Year', title="平均価格",
                                  color_discrete_map={'2018': '#95a5a6', '2025': '#e74c3c'})
                fig_price.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
                st.plotly_chart(fig_price, use_container_width=True)
            else:
                st.warning("価格データが不足しています")

# ==================== モード3: モデル間比較 ====================
else:  # モデル間比較
    st.title("⚔️ モデル間比較 (2018 vs 2025)")
    
    st.sidebar.markdown("### 比較するモデルを選択")
    
    # モデル1選択
    brand1 = st.sidebar.selectbox("ブランド1", ["Honda", "Toyota"], key='brand1')
    models1 = HONDA_MAJOR if brand1 == "Honda" else TOYOTA_MAJOR
    model1 = st.sidebar.selectbox("モデル1", models1, key='model1')
    
    # モデル2選択
    brand2 = st.sidebar.selectbox("ブランド2", ["Honda", "Toyota"], key='brand2')
    models2 = HONDA_MAJOR if brand2 == "Honda" else TOYOTA_MAJOR
    model2 = st.sidebar.selectbox("モデル2", models2, key='model2')
    
    # データ抽出（部分一致）
    df1 = df[df['Brand_P'] == brand1].copy()
    df1 = df1[df1['Model_P'].str.contains(model1, case=False, na=False)]
    df1['Model_Label'] = f"{brand1} {model1}"
    
    df2 = df[df['Brand_P'] == brand2].copy()
    df2 = df2[df2['Model_P'].str.contains(model2, case=False, na=False)]
    df2['Model_Label'] = f"{brand2} {model2}"
    
    compare_df = pd.concat([df1, df2])
    
    st.markdown(f"## {brand1} {model1} vs {brand2} {model2}")
    
    # タブ構成
    tab1, tab2, tab3 = st.tabs(["📊 基本指標", "👤 顧客属性", "💰 価格・地域"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 離反率比較")
            st.caption(f"💡 離反率分析には **{WEIGHT_LOYALTY}** を使用")
            # Detailed_Statusを使って離反率を計算
            defection_data = []
            for model_label in compare_df['Model_Label'].unique():
                for year in ['2018', '2025']:
                    model_year_df = compare_df[(compare_df['Model_Label'] == model_label) & (compare_df['Year'] == year)]
                    if len(model_year_df) > 0:
                        total_weight = model_year_df[WEIGHT_LOYALTY].sum()
                        defection_weight = model_year_df[model_year_df['Detailed_Status'].str.contains('Defection', case=False, na=False)][WEIGHT_LOYALTY].sum()
                        defection_rate = (defection_weight / total_weight * 100) if total_weight > 0 else 0
                        defection_data.append({'Model_Label': model_label, 'Year': year, 'Share (%)': defection_rate})
            defection_stats = pd.DataFrame(defection_data)
            
            fig_def = px.bar(defection_stats, x='Model_Label', y='Share (%)', color='Year',
                            barmode='group', text='Share (%)',
                            color_discrete_map={'2018': '#95a5a6', '2025': '#e74c3c'})
            fig_def.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
            st.plotly_chart(fig_def, use_container_width=True)
        
        with col2:
            st.markdown("### ボリューム比較")
            st.caption(f"💡 ボリューム分析には **{WEIGHT_DISTRIBUTION}** を使用")
            vol = compare_df.groupby(['Model_Label', 'Year'])[WEIGHT_DISTRIBUTION].sum().reset_index()
            fig_vol = px.bar(vol, x='Model_Label', y=WEIGHT_DISTRIBUTION, color='Year',
                            barmode='group', text=WEIGHT_DISTRIBUTION, title="サンプルボリューム")
            fig_vol.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
            st.plotly_chart(fig_vol, use_container_width=True)
        
        with col3:
            st.markdown("### 平均価格比較")
            price_data = []
            for (model, year), group in compare_df.groupby(['Model_Label', 'Year']):
                try:
                    # Price列がNaNでなく、かつ0より大きいレコードのみを使用
                    valid_price = group[(group['Price'].notna()) & (group['Price'] > 0)].copy()
                    if len(valid_price) > 0:
                        avg_price = (valid_price['Price'].astype(float) * valid_price[WEIGHT_DISTRIBUTION]).sum() / valid_price[WEIGHT_DISTRIBUTION].sum()
                        price_data.append({'Model_Label': model, 'Year': year, 'Avg_price': avg_price})
                except:
                    pass
            
            if len(price_data) > 0:
                price_avg = pd.DataFrame(price_data)
                
                fig_price = px.bar(price_avg, x='Model_Label', y='Avg_Price', color='Year',
                                  barmode='group', text='Avg_Price',
                                  color_discrete_map={'2018': '#95a5a6', '2025': '#e74c3c'})
                fig_price.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
                st.plotly_chart(fig_price, use_container_width=True)
            else:
                st.warning("価格データが不足しています")
    
    with tab2:
        st.markdown("### 👤 顧客属性比較")
        
        demo_attr = st.selectbox("属性", ["Income", "Age", "Lifestage"], key='compare_demo')
        
        # Ageの場合は年齢グループ化を適用
        if demo_attr == "Age":
            # 年齢グループ化関数
            def group_age_compare(age_val):
                if pd.isna(age_val):
                    return None
                age_str = str(age_val)
                
                if 'Under 20' in age_str or age_str in ['16.0', '17.0', '18.0', '19.0']:
                    return '<20'
                elif '70 Or Over' in age_str:
                    return '70+'
                elif ' To ' in age_str:
                    parts = age_str.split()
                    if len(parts) >= 3:
                        return f'{parts[0]}-{parts[2]}'
                
                try:
                    age_num = float(age_str.replace('.0', ''))
                    if age_num < 20:
                        return '<20'
                    elif age_num >= 70:
                        return '70+'
                    else:
                        lower = int(age_num // 5 * 5)
                        upper = lower + 4
                        return f'{lower}-{upper}'
                except:
                    return None
            
            # 年齢グループ化したデータを作成
            age_grouped_df = compare_df.copy()
            age_grouped_df['Age_Group'] = age_grouped_df['Age'].apply(group_age_compare)
            age_grouped_df = age_grouped_df[age_grouped_df['Age_Group'].notna()]
            
            # 年齢グループの順序を定義
            age_order = ['<20', '20-24', '25-29', '30-34', '35-39', '40-44',
                        '45-49', '50-54', '55-59', '60-64', '65-69', '70+']
            
            # グループ化して集計
            demo_stats = get_weighted_share(age_grouped_df, ['Model_Label', 'Year'], 'Age_Group')
            
            # 2018年と2025年を並べて表示
            fig = make_subplots(rows=1, cols=2, subplot_titles=("2018年", "2025年"))
            
            for i, year in enumerate(['2018', '2025'], 1):
                year_data = demo_stats[demo_stats['Year'] == year]
                
                for model_label in year_data['Model_Label'].unique():
                    model_data = year_data[year_data['Model_Label'] == model_label]
                    # 年齢グループの順序でソート
                    model_data['Age_Group'] = pd.Categorical(model_data['Age_Group'],
                                                             categories=age_order,
                                                             ordered=True)
                    model_data = model_data.sort_values('Age_Group')
                    
                    fig.add_trace(
                        go.Bar(x=model_data['Age_Group'], y=model_data['Share (%)'],
                               name=model_label, legendgroup=model_label, showlegend=(i==1)),
                        row=1, col=i
                    )
            
            fig.update_layout(height=500, barmode='group', title_text="年齢グループ 構成比較")
            fig.update_xaxes(categoryorder='array', categoryarray=age_order)
            st.plotly_chart(fig, use_container_width=True)
        elif demo_attr == "Income":
            # Incomeの場合は年収グループ化を適用
            income_grouped_df = compare_df.copy()
            income_grouped_df['Income_Group'] = income_grouped_df['Income'].apply(group_income_for_matrix)
            income_grouped_df = income_grouped_df[income_grouped_df['Income_Group'].notna()]
            
            # 年収グループの順序を定義（低い順）
            income_order = ['<$50k', '$50-99k', '$100-149k', '$150-199k',
                           '$200-299k', '$300-399k', '$400-499k', '$500k+']
            
            # グループ化して集計
            demo_stats = get_weighted_share(income_grouped_df, ['Model_Label', 'Year'], 'Income_Group')
            
            # 2018年と2025年を並べて表示
            fig = make_subplots(rows=1, cols=2, subplot_titles=("2018年", "2025年"))
            
            for i, year in enumerate(['2018', '2025'], 1):
                year_data = demo_stats[demo_stats['Year'] == year]
                
                for model_label in year_data['Model_Label'].unique():
                    model_data = year_data[year_data['Model_Label'] == model_label]
                    # 年収グループの順序でソート
                    model_data['Income_Group'] = pd.Categorical(model_data['Income_Group'],
                                                                categories=income_order,
                                                                ordered=True)
                    model_data = model_data.sort_values('Income_Group')
                    
                    fig.add_trace(
                        go.Bar(x=model_data['Income_Group'], y=model_data['Share (%)'],
                               name=model_label, legendgroup=model_label, showlegend=(i==1)),
                        row=1, col=i
                    )
            
            fig.update_layout(height=500, barmode='group', title_text="年収グループ 構成比較")
            fig.update_xaxes(categoryorder='array', categoryarray=income_order)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Lifestageなど他の属性は既存の処理
            demo_stats = get_weighted_share(compare_df, ['Model_Label', 'Year'], demo_attr)
            
            # 2018年と2025年を並べて表示
            fig = make_subplots(rows=1, cols=2, subplot_titles=("2018年", "2025年"))
            
            for i, year in enumerate(['2018', '2025'], 1):
                year_data = demo_stats[demo_stats['Year'] == year]
                
                for model_label in year_data['Model_Label'].unique():
                    model_data = year_data[year_data['Model_Label'] == model_label]
                    fig.add_trace(
                        go.Bar(x=model_data[demo_attr], y=model_data['Share (%)'],
                               name=model_label, legendgroup=model_label, showlegend=(i==1)),
                        row=1, col=i
                    )
            
            fig.update_layout(height=500, barmode='group', title_text=f"{demo_attr} 構成比較")
            st.plotly_chart(fig, use_container_width=True)
        
        # Incomeが選択された場合のみ、年齢×年収マトリクスを追加表示
        if demo_attr == "Income":
            st.divider()
            st.markdown("### 📊 年齢×年収マトリクス（モデル別）")
            st.caption("各モデルの顧客の年齢層と年収層の分布を割合(%)で可視化")
            
            # データ準備
            matrix_df_compare = compare_df[compare_df['Age'].notna() & compare_df['Income'].notna()].copy()
            
            if len(matrix_df_compare) > 0:
                # 年齢・年収グループ化関数（モデル別分析と同じ）
                def group_age_matrix_cmp(age_val):
                    if pd.isna(age_val):
                        return None
                    age_str = str(age_val)
                    if 'Under 20' in age_str or age_str in ['16.0', '17.0', '18.0', '19.0']:
                        return '<20'
                    elif '70 Or Over' in age_str:
                        return '70+'
                    elif ' To ' in age_str:
                        parts = age_str.split()
                        if len(parts) >= 3:
                            return f'{parts[0]}-{parts[2]}'
                    try:
                        age_num = float(age_str.replace('.0', ''))
                        if age_num < 20:
                            return '<20'
                        elif age_num >= 70:
                            return '70+'
                        else:
                            lower = int(age_num // 5 * 5)
                            upper = lower + 4
                            return f'{lower}-{upper}'
                    except:
                        return None
                
                # グループ化を適用
                matrix_df_compare['Age_Group'] = matrix_df_compare['Age'].apply(group_age_matrix_cmp)
                matrix_df_compare['Income_Group'] = matrix_df_compare['Income'].apply(group_income_for_matrix)
                
                # マトリクス作成関数
                def create_compare_matrix(data, year_val):
                    year_data = data[data['Year'] == year_val]
                    year_data = year_data[year_data['Age_Group'].notna() & year_data['Income_Group'].notna()]
                    
                    pivot = year_data.pivot_table(
                        values=WEIGHT_DISTRIBUTION,
                        index='Income_Group',
                        columns='Age_Group',
                        aggfunc='sum',
                        fill_value=0
                    )
                    
                    total = pivot.sum().sum()
                    if total > 0:
                        pivot = (pivot / total) * 100
                    
                    age_order = ['<20', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                                '50-54', '55-59', '60-64', '65-69', '70+']
                    income_order = ['$500k+', '$400-499k', '$300-399k', '$200-299k',
                                   '$150-199k', '$100-149k', '$50-99k', '<$50k']
                    
                    existing_ages = [a for a in age_order if a in pivot.columns]
                    existing_incomes = [i for i in income_order if i in pivot.index]
                    
                    pivot = pivot.reindex(index=existing_incomes, columns=existing_ages, fill_value=0)
                    return pivot
                
                # 各モデルのマトリクスを表示
                for model_label in compare_df['Model_Label'].unique():
                    st.markdown(f"#### {model_label}")
                    model_data = matrix_df_compare[matrix_df_compare['Model_Label'] == model_label]
                    
                    if len(model_data) > 0:
                        matrix_2018 = create_compare_matrix(model_data, '2018')
                        matrix_2025 = create_compare_matrix(model_data, '2025')
                        
                        # 軸を統一
                        all_ages = matrix_2018.columns.union(matrix_2025.columns)
                        all_incomes = matrix_2018.index.union(matrix_2025.index)
                        matrix_2018 = matrix_2018.reindex(index=all_incomes, columns=all_ages, fill_value=0)
                        matrix_2025 = matrix_2025.reindex(index=all_incomes, columns=all_ages, fill_value=0)
                        
                        # 差分計算
                        matrix_diff = matrix_2025 - matrix_2018
                        
                        # 3つのヒートマップを表示
                        col_m1, col_m2, col_m3 = st.columns(3)
                        
                        with col_m1:
                            st.markdown("**2018年**")
                            vmin = matrix_2018.values[matrix_2018.values > 0].min() if (matrix_2018.values > 0).any() else 0
                            vmax = np.percentile(matrix_2018.values[matrix_2018.values > 0], 95) if (matrix_2018.values > 0).any() else matrix_2018.values.max()
                            fig = px.imshow(matrix_2018, labels=dict(x="年齢", y="年収", color="割合 (%)"),
                                           aspect="auto", color_continuous_scale="RdBu_r",
                                           zmin=vmin, zmax=vmax, text_auto='.1f')
                            fig.update_layout(height=350)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_m2:
                            st.markdown("**2025年**")
                            vmin = matrix_2025.values[matrix_2025.values > 0].min() if (matrix_2025.values > 0).any() else 0
                            vmax = np.percentile(matrix_2025.values[matrix_2025.values > 0], 95) if (matrix_2025.values > 0).any() else matrix_2025.values.max()
                            fig = px.imshow(matrix_2025, labels=dict(x="年齢", y="年収", color="割合 (%)"),
                                           aspect="auto", color_continuous_scale="RdBu_r",
                                           zmin=vmin, zmax=vmax, text_auto='.1f')
                            fig.update_layout(height=350)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_m3:
                            st.markdown("**差分 (2025 - 2018)**")
                            abs_max = np.percentile(np.abs(matrix_diff.values), 95)
                            fig = px.imshow(matrix_diff, labels=dict(x="年齢", y="年収", color="変化量 (%)"),
                                           aspect="auto", color_continuous_scale="RdBu_r",
                                           zmin=-abs_max, zmax=abs_max, color_continuous_midpoint=0, text_auto='.1f')
                            fig.update_layout(height=350)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.divider()
                    else:
                        st.warning(f"{model_label}の年齢×年収データが不足しています")
            else:
                st.warning("年齢×年収データが不足しています")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 居住地特性")
            urb_stats = get_weighted_share(compare_df, ['Model_Label', 'Year'], 'Urbanicity')
            
            fig_urb = px.bar(urb_stats, x='Urbanicity', y='Share (%)',
                            color='Model_Label', barmode='group',
                            facet_col='Year', title="居住地特性の比較",
                            category_orders={'Year': ['2018', '2025']})
            st.plotly_chart(fig_urb, use_container_width=True)
        
        with col2:
            st.markdown("### 支払方法")
            pay_stats = get_weighted_share(compare_df, ['Model_Label', 'Year'], 'Payment_Method')
            
            fig_pay = px.bar(pay_stats, x='Payment_Method', y='Share (%)',
                            color='Model_Label', barmode='group',
                            facet_col='Year', title="支払方法の比較",
                            category_orders={'Year': ['2018', '2025']})
            st.plotly_chart(fig_pay, use_container_width=True)

st.divider()
st.caption("Honda vs Toyota Evolution Analysis | Data Source: Strategic Vision NVES 2018 & 2025")
