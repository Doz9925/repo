import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import random


df_customer = pd.read_csv("df_final3.csv")
df_charge = pd.read_csv("DS_Charge_data.csv", encoding='cp949')

image_path = "image11.png"


df_customer['이용서비스수'] = (
    df_customer[['보안서비스', '백업서비스', '기술지원서비스']]
    .apply(lambda x: x.map({'Yes': 1, 'No': 0}))
    .sum(axis=1))



df2 = df_customer[['연령', '결혼여부', '부양자유무', '추천횟수', '영수증발급여부', '과금방식', 
                   '보안서비스', '백업서비스', '기술지원서비스', '데이터무제한', '데이터사용량', 'LTV2']]


yes_no_columns = ['결혼여부', '부양자유무', '영수증발급여부', '보안서비스', '백업서비스', '기술지원서비스', '데이터무제한']
for col in yes_no_columns:
    df2[col] = df2[col].map({'Yes': 1, 'No': 0})

df2['과금방식'] = df2['과금방식'].map({'신용카드': 0, '계좌이체': 1, '이체/메일확인': 2})


scaler = StandardScaler()
df2_scaled = scaler.fit_transform(df2)


kmeans = KMeans(n_clusters=3, random_state=1234)
df_customer['Cluster'] = kmeans.fit_predict(df2_scaled)


pca = PCA(n_components=3)
df2_pca_3d = pca.fit_transform(df2_scaled)


df2_pca_3d_df = pd.DataFrame(df2_pca_3d, columns=['PCA1', 'PCA2', 'PCA3'])
df2_pca_3d_df['고객ID'] = df_customer['고객ID']
df2_pca_3d_df['Cluster'] = df_customer['Cluster'].astype(str)


with open('C:\\Users\\cjswo\\Desktop\\Untitled Folder\\best_model11.sav', 'rb') as model_file:
    best_model = pickle.load(model_file)

# Streamlit
st.set_page_config(page_title="고객 이탈 예측 대시보드", layout="wide")
st.markdown("<h1 style='text-align: center; font-size: 3em; font-weight: bold; color: white; margin-top: 20px;'>고객 이탈 예측 대시보드</h1>", unsafe_allow_html=True)







def generate_color():
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    return random.choice(colors)



st.sidebar.image(image_path, use_column_width=True)
input_ids = st.sidebar.text_input("고객 ID를 입력하세요 (여러 ID는 쉼표로 구분):")


if input_ids:
    customer_ids = [id.strip() for id in input_ids.split(",")]

    for customer_id in customer_ids:
        st.markdown(f"<h2 style='text-align: left; font-size: 2em; color: white; margin-top: 10px;'>고객 ID: {customer_id}</h2>", unsafe_allow_html=True)
        customer_color = generate_color()


        column1, column2, column3 = st.columns([2, 2, 1])

        # Column 1: 고객 이탈 정보 섹션
        with column1:
            st.markdown("<h3 style='font-size: 2em; font-weight: bold; color: white;'>고객 이탈 정보</h3>", unsafe_allow_html=True)
            st.markdown(f"<hr style='border:2px solid {customer_color};margin-top: 5px;'/>", unsafe_allow_html=True)


            customer_info = df_customer[df_customer['고객ID'] == customer_id]
            if not customer_info.empty:
                customer_data = customer_info[['연령', '결혼여부', '부양자유무', '추천횟수',
                                               '영수증발급여부', '과금방식', '보안서비스', '백업서비스', '기술지원서비스', '데이터무제한', '데이터사용량', 'LTV2']]
                churn_prediction = best_model.predict(customer_data)[0]

   
                if churn_prediction == 1:
                    status_text = "이탈 고객"
                    status_color = "red"
                else:
                    status_text = "안전 고객"
                    status_color = "blue"
                st.markdown(f"<h2 style='color:{status_color};'>{status_text}</h2>", unsafe_allow_html=True)

                prob_1yr = customer_info['1년이후잔존확률'].values[0] * 100
                prob_2yr = customer_info['2년이후잔존확률'].values[0] * 100


                prob_data_1yr = [prob_1yr, 100 - prob_1yr]
                prob_data_2yr = [prob_2yr, 100 - prob_2yr]
                labels = ["잔존 확률", "이탈 확률"]

    
                pie_col1, pie_col2 = st.columns(2)


                with pie_col1:
                    fig_1yr = px.pie(
                        names=labels,
                        values=prob_data_1yr,
                        color=labels,
                        color_discrete_map={"잔존 확률": "green", "이탈 확률": "red"}
                    )
                    fig_1yr.update_traces(hole=0.4, textinfo='percent+label')
                    fig_1yr.update_layout(title="1년 이후 잔존 확률", width=300, height=300)
                    st.plotly_chart(fig_1yr, use_container_width=True)


                with pie_col2:
                    fig_2yr = px.pie(
                        names=labels,
                        values=prob_data_2yr,
                        color=labels,
                        color_discrete_map={"잔존 확률": "green", "이탈 확률": "red"}
                    )
                    fig_2yr.update_traces(hole=0.4, textinfo='percent+label')
                    fig_2yr.update_layout(title="2년 이후 잔존 확률", width=300, height=300)
                    st.plotly_chart(fig_2yr, use_container_width=True)

                st.markdown(f"<hr style='border:2px solid {customer_color};margin-top: 14px;'/>", unsafe_allow_html=True)


                cluster_label = customer_info['Cluster'].values[0]
                st.subheader(f"군집: {cluster_label}")

  
                fig = px.scatter_3d(df2_pca_3d_df, x='PCA1', y='PCA2', z='PCA3', 
                                    color='Cluster', 
                                    category_orders={'Cluster': ['0', '1', '2']},
                                    color_discrete_map={'0': 'navy', '1': 'green', '2': 'yellow'},
                                    title='3D PCA KMeans Clustering Visualization')
                fig.update_traces(marker=dict(size=3, opacity=0.1))


                customer_pca_info = df2_pca_3d_df[df2_pca_3d_df['고객ID'] == customer_id]
                fig.add_trace(go.Scatter3d(
                    x=[customer_pca_info['PCA1'].values[0]],
                    y=[customer_pca_info['PCA2'].values[0]],
                    z=[customer_pca_info['PCA3'].values[0]],
                    mode='markers+text',
                    marker=dict(size=8, color='red', symbol='x'),
                    text=['고객 위치'],
                    textposition='top center'
                ))
                st.plotly_chart(fig, use_container_width=True)

            # Column 2: 월별 과금 정보 섹션
            with column2:
                st.markdown("<h3 style='font-size: 2em; font-weight: bold; color: white;'>월별 과금액</h3>", unsafe_allow_html=True)
                st.markdown(f"<hr style='border:2px solid {customer_color};margin-top: 5px;'/>", unsafe_allow_html=True)

                monthly_charges = df_charge[df_charge['고객ID'] == customer_id][['과금일', '과금액']]
                if not monthly_charges.empty:
                    total_charge = monthly_charges['과금액'].sum()
                    avg_charge = monthly_charges['과금액'].mean()
                    charge_count = monthly_charges.shape[0]

                    st.markdown(f"<h4>과금 횟수: <span style='color:{customer_color};'>{charge_count} 회</span></h4>", unsafe_allow_html=True)
                    st.markdown(f"<h4>총 과금액: <span style='color:{customer_color};'>{total_charge:.2f} 원</span></h4>", unsafe_allow_html=True)
                    st.markdown(f"<h4>평균 과금액: <span style='color:{customer_color};'>{avg_charge:.2f} 원</span></h4>", unsafe_allow_html=True)
                    st.markdown(f"<hr style='border:2px solid {customer_color};'/>", unsafe_allow_html=True)


                    fig = px.line(monthly_charges, x='과금일', y='과금액', title=f"{customer_id} 월별 과금액 변화", line_shape='spline')
                    fig.update_traces(line=dict(color=customer_color), marker=dict(size=8))
                    fig.update_layout(title={'text': f"{customer_id} 월별 과금액 변화", 'x':0.5, 'xanchor': 'center'},
                                      title_font=dict(size=24) ,width=340, height=350)
                    fig.add_scatter(x=monthly_charges['과금일'], y=monthly_charges['과금액'], mode='markers', marker=dict(size=6, color=customer_color))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("해당 고객의 과금 데이터가 없습니다.")

            # Column 3: 고객 정보 섹션
            with column3:
                st.markdown("<h3 style='font-size: 2em; font-weight: bold; color: white;'>고객 정보</h3>", unsafe_allow_html=True)
                st.markdown(f"<hr style='border:2px solid {customer_color};margin-top: 5px;'/>", unsafe_allow_html=True)

                if not customer_info.empty:

                    customer_info_display = customer_info[['성별', '연령', '결혼여부', '부양자수', '추천횟수',
                                                        '과금방식', '데이터무제한', '데이터사용량', '시작일', '종료일',
                                                        '고객이탈여부', '이탈유형', '만족도', '로밍사용료']]
                    customer_info_display['LTV_new'] = customer_info['LTV2']
                    customer_info_display.fillna("-", inplace=True)
                    st.dataframe(customer_info_display.T, use_container_width=True)


                    st.markdown("<h3 style='font-size: 2em; font-weight: bold; color: white; margin-top: 7px;'>서비스 유무</h3>", unsafe_allow_html=True)
                    st.markdown(f"<hr style='border:2px solid {customer_color};margin-top: 5px;'/>", unsafe_allow_html=True)


                    service_info = customer_info[['보안서비스', '백업서비스', '기술지원서비스']]
                    service_info.fillna("-", inplace=True)
                    st.dataframe(service_info.T, use_container_width=True)
