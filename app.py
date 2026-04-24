import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="ML Vendas", page_icon="🛒", layout="wide")

# =========================
# LOAD
# =========================
@st.cache_resource
def carregar_modelo():
    model = joblib.load("model/modelo.pkl")
    encoder = joblib.load("model/encoder.pkl")
    return model, encoder

@st.cache_data
def carregar_dados():
    df = pd.read_csv("data/dados.csv")

    meses_map = {
        "janeiro":1,"fevereiro":2,"marco":3,"abril":4,
        "maio":5,"junho":6,"julho":7,"agosto":8,
        "setembro":9,"outubro":10,"novembro":11,"dezembro":12
    }

    df["mes_num"] = df["Mes"].str.lower().map(meses_map)
    df["data"] = pd.to_datetime(df["Ano"].astype(str) + "-" + df["mes_num"].astype(str) + "-01")

    return df

model, encoder = carregar_modelo()
df = carregar_dados()
redes = list(encoder.classes_)

meses_nome = {
    1:"Janeiro",2:"Fevereiro",3:"Marco",4:"Abril",
    5:"Maio",6:"Junho",7:"Julho",8:"Agosto",
    9:"Setembro",10:"Outubro",11:"Novembro",12:"Dezembro"
}

# =========================
# UI
# =========================
st.title("🛒 ML Vendas - Supermercado")
st.divider()

tab1, tab2, tab3 = st.tabs(["Previsao", "Forecast", "Historico"])

# =========================
# TAB 1
# =========================
with tab1:
    st.subheader("Previsao para um periodo especifico")

    col1, col2, col3, col4 = st.columns(4)

    rede = col1.selectbox("Rede", redes)

    mes = col2.selectbox("Mes", list(meses_nome.values()))
    mes_num = list(meses_nome.keys())[list(meses_nome.values()).index(mes)]

    ano = col3.number_input("Ano", 2024, 2035, 2025)

    cupons = col4.number_input("Qtd Cupons", 1000, 5000000, 600000, step=10000)

    if st.button("Prever Venda", use_container_width=True):

        try:
            rede_enc = encoder.transform([rede])[0]

            entrada = pd.DataFrame({
                "Rede_enc": [rede_enc],
                "mes_num": [mes_num],
                "Ano": [ano],
                "Qtd Vendas (cupons)": [cupons]
            })

            # GARANTE ORDEM
            entrada = entrada[["Rede_enc","mes_num","Ano","Qtd Vendas (cupons)"]]

            pred = model.predict(entrada)[0]
            ticket = pred / cupons if cupons > 0 else 0

            c1, c2, c3 = st.columns(3)
            c1.metric("Previsao de Venda", f"R$ {pred:,.2f}")
            c2.metric("Ticket Medio", f"R$ {ticket:.2f}")
            c3.metric("Cupons", f"{cupons:,}")

        except Exception as e:
            st.error(f"Erro ao prever: {e}")

# =========================
# TAB 2 - FORECAST
# =========================
with tab2:
    st.subheader("Forecast - proximos meses")

    col1, col2, col3 = st.columns(3)

    rede_f = col1.selectbox("Rede", redes, key="rede_forecast")
    meses_ahead = col2.slider("Meses para prever", 1, 12, 6)
    cupons_f = col3.number_input("Qtd Cupons media", 1000, 5000000, 600000, step=10000)

    if st.button("Gerar Forecast", use_container_width=True):

        with st.spinner("Gerando previsões..."):

            df_rede = df[df["Rede"] == rede_f].sort_values("data")

            if df_rede.empty:
                st.warning("Sem dados históricos para essa rede.")
                st.stop()

            rede_enc_f = encoder.transform([rede_f])[0]

            historico = df_rede[["data","Vlr Venda"]].rename(columns={"Vlr Venda":"valor"})
            last_date = historico["data"].max()

            previsoes = []

            for i in range(1, meses_ahead + 1):
                next_date = last_date + pd.DateOffset(months=i)

                entrada = pd.DataFrame({
                    "Rede_enc": [rede_enc_f],
                    "mes_num": [next_date.month],
                    "Ano": [next_date.year],
                    "Qtd Vendas (cupons)": [cupons_f]
                })

                entrada = entrada[["Rede_enc","mes_num","Ano","Qtd Vendas (cupons)"]]

                try:
                    pred = model.predict(entrada)[0]
                except Exception as e:
                    st.error(f"Erro no modelo: {e}")
                    st.stop()

                previsoes.append({"data": next_date, "valor": pred})

            df_prev = pd.DataFrame(previsoes)

            # GRÁFICO
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=historico["data"],
                y=historico["valor"],
                mode="lines+markers",
                name="Historico"
            ))

            fig.add_trace(go.Scatter(
                x=pd.concat([historico["data"].tail(1), df_prev["data"]]),
                y=pd.concat([historico["valor"].tail(1), df_prev["valor"]]),
                mode="lines+markers",
                name="Forecast",
                line=dict(dash="dot")
            ))

            fig.update_layout(
                template="plotly_dark",
                title=f"Forecast - {rede_f}",
                xaxis_title="Periodo",
                yaxis_title="Venda (R$)",
                height=450,
                hovermode="x unified"
            )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_prev, use_container_width=True, hide_index=True)


# =========================
# TAB 3
# =========================
with tab3:
    st.subheader("Historico de vendas por rede")

    fig2 = px.line(
        df.sort_values("data"),
        x="data",
        y="Vlr Venda",
        color="Rede",
        markers=True,
        template="plotly_dark"
    )

    fig2.update_layout(height=450, hovermode="x unified")

    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df, use_container_width=True, hide_index=True)