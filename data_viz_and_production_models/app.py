import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

st.set_page_config(page_title="Tech Challenge", page_icon="💻")

st.header("[Tech Challenge] Data Viz and Production Models")
st.subheader("Análise de dados do preço do barril de petróleo")


@st.cache_data
def ler_dados():
    # Importando os dados da tabela HTML
    df = pd.read_html(
        "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"
    )[2]

    # Renomeando colunas
    df = df.rename(columns={0: "data", 1: "preco"})
    df = df.iloc[1:]

    # Formatando preço
    df["preco"] = df["preco"].astype("float")
    df["preco"] = df["preco"] / 100

    # Formatando data
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")

    # Separando por ano e mês
    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month

    return df


df = ler_dados()

# Seleção de anos
anos = [2019, 2020, 2021, 2022, 2023]
st.sidebar.title("Filtros")
anos = st.sidebar.multiselect("Selecione os anos", df["ano"].unique(), default=anos)
meses = st.sidebar.slider("Previsão em Meses", min_value=1, max_value=12, value=12)

# Filtrando dados
df_filtrado = df[df["ano"].isin(anos)]

# Agrupando por ano e mês para calcular a média de preço
df_media = df_filtrado.groupby(["ano", "mes"]).agg({"preco": "mean"}).reset_index()

# Criando o gráfico de preço histórico do barril de petróleo
fig = px.line(
    df_filtrado,
    x="data",
    y="preco",
    labels={"data": "Data", "preco": "Preço"},
    title="Preço histórico do barril de petróleo (USD)",
)
st.plotly_chart(fig, use_container_width=True)

# Criando o gráfico de preço médio mensal do barril de petróleo por ano
fig = px.line(
    df_media,
    x="mes",
    y="preco",
    color="ano",
    labels={"mes": "Mês", "preco": "Preço médio"},
    title="Preço médio mensal do barril de petróleo (USD) por ano",
)
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    """
    Nestes gráficos, é possível notar algumas tendências e eventos que podem ser relacionados a contextos geopolíticos e econômicos, incluindo a pandemia de COVID-19:
    
    1. *Volatilidade em 2020*: Observa-se uma significativa volatilidade no ano de 2020, o que poderia estar relacionado ao início da pandemia de COVID-19. A pandemia teve um impacto profundo na demanda global por petróleo devido a bloqueios e uma redução significativa nas viagens, o que poderia explicar a queda e as flutuações nos preços.
    2. *Recuperação em 2021*: O aumento dos preços em 2021 pode indicar uma recuperação à medida que as vacinas começaram a ser distribuídas e os países começaram a reabrir suas economias. Isso provavelmente levou a um aumento na demanda por petróleo, refletindo-se em uma tendência de alta no preço.
    3. *Estabilização em 2022*: Em 2022, parece haver uma estabilização ou mesmo uma ligeira tendência de declínio nos preços. Isso pode ser um sinal de que os mercados estão começando a se ajustar ao "novo normal" com demandas e ofertas de petróleo se estabilizando após as turbulências iniciais da pandemia.
    4. *Tendência em 2023*: A continuidade da tendência em 2023 poderia sugerir uma estabilidade no mercado de petróleo ou a ausência de eventos significativos de perturbação, mas isso dependeria da situação geopolítica atual, como a estabilidade de regiões produtoras de petróleo, políticas de produção da OPEP, e outras tensões internacionais.
    """
)

st.subheader("Previsão do preço do barril de petróleo")


@st.cache_data
def criar_previsao(df, periods=12):
    # Criando o modelo Prophet
    modelo = Prophet(daily_seasonality=True)
    modelo.fit(df[["data", "preco"]].rename(columns={"data": "ds", "preco": "y"}))

    # Criando dataframe de previsão
    futuro = modelo.make_future_dataframe(periods=periods, freq="M")
    previsao = modelo.predict(futuro)

    # Criando a figura Plotly
    fig = go.Figure()

    # Adicionando série original
    fig.add_trace(
        go.Scatter(x=df["data"], y=df["preco"], mode="lines", name="Dados Originais")
    )

    # Adicionando a previsão
    fig.add_trace(
        go.Scatter(x=previsao["ds"], y=previsao["yhat"], mode="lines", name="Previsão")
    )

    # Adicionando intervalos de incerteza
    fig.add_trace(
        go.Scatter(
            x=previsao["ds"],
            y=previsao["yhat_upper"],
            fill=None,
            mode="lines",
            line=dict(color="gray"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=previsao["ds"],
            y=previsao["yhat_lower"],
            fill="tonexty",
            mode="lines",
            line=dict(color="gray"),
            showlegend=False,
        )
    )

    # Atualizando layouts
    fig.update_layout(
        title="Previsão de Preço do Barril de Petróleo com Prophet",
        xaxis_title="Data",
        yaxis_title="Preço",
        legend_title="Legenda",
    )

    return fig


fig = criar_previsao(df=df_filtrado, periods=meses)
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    """
    Como é possível observar, a tendência para 2024 é de um aumento gradual no preço do barril de petróleo, com uma faixa de incerteza que se amplia ao longo do tempo. Isso sugere que, embora haja uma tendência de alta, a volatilidade dos preços pode aumentar à medida que nos afastamos do presente.
    """
)
