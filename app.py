import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Настройка страницы
st.set_page_config(
    page_title="Оптимизация интервалов испытания скважин",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("🛢️ Оптимизация интервалов испытания нефтяных скважин")
st.markdown("---")

# Функции оптимизации (перенесены из Jupyter notebook)
def optimize_interval(
    df, 
    well_name,
    mu_o=0.43,  # вязкость нефти, Па·с
    mu_w=0.33,  # вязкость воды, Па·с
    P_res=30 * 1.013e5,  # пластовое давление, Па
    P_well=20 * 1.013e5, # давление на устье, Па
    r_w=0.1,             # радиус скважины, м
    r_e=500,             # радиус дренирования, м
    h=0.1,               # толщина пласта, м
    Soi=0.28,            # остаточная нефтеёмкость
    B=1.2,               # объемный коэффициент
    prise_oil=15000,     # цена нефти, руб/м3
    prise_w=800,         # утилизация воды, руб/м3
    opex_liquid=250,     # OPEX на жидкость, руб/м3
    ndpi_rate=0.01,      # НДПИ от выручки
    reculc=351182.040,      # коэффициент пересчёта
    window_sizes_min_m=4,
    window_sizes_max_m=35
):
    """
    Поиск оптимального интервала испытания по прибыли
    """
    # фильтрация по скважине
    well = df[df.well == well_name].copy()

    # фазовые параметры
    well['o'] = (well['fill_cb_res'] * (1 - well['KVS_fin']) - Soi) / mu_o
    well['o'] = np.clip(well['o'], 0, 1)
    o = well['o']
    kv = 1 - well['fill_cb_res'] * (1 - well['KVS_fin'])
    w = (kv - well['KVS_fin']) / mu_w

    well['f_o'] = o / (o + w)
    well['Q_total'] = (
        (2 * np.pi * well["KVS_fin"] * h * (P_res - P_well)) /
        (B * np.log(r_e / r_w))
    )
    well['Q_oil'] = well['Q_total'] * well['f_o']
    well['Q_wat'] = well['Q_total'] * (1 - well['f_o'])

    # перевод метров в точки (шаг 0.1 м → умножаем на 10)
    window_sizes_min = window_sizes_min_m * 10
    window_sizes_max = window_sizes_max_m * 10

    best_window = None
    best_start = None
    best_metric = -np.inf
    best_result = {}

    for window in range(window_sizes_min, window_sizes_max + 1):
        for start in range(0, len(well) - window + 1):
            end = start + window
            df_window = well.iloc[start:end]

            # выручка
            revenue = prise_oil * df_window['Q_oil'].sum() / reculc
            # налог
            ndpi = ndpi_rate * revenue
            # затраты (вода + подъём жидкости)
            costs = (
                prise_w * df_window['Q_wat'].sum() / reculc +
                opex_liquid * df_window['Q_total'].sum() / reculc
            )
            # чистая прибыль
            profit = revenue - ndpi - costs

            if profit > best_metric:
                best_metric = profit
                best_window = window
                best_start = start
                best_result = {
                    'revenue': revenue,
                    'ndpi': ndpi,
                    'costs': costs,
                    'profit': profit,
                    'Q_oil': df_window['Q_oil'].sum() / reculc,
                    'Q_wat': df_window['Q_wat'].sum() / reculc,
                    'depth_start': df_window.iloc[0]['DEPTH'],
                    'depth_end': df_window.iloc[-1]['DEPTH'],
                    'window_m': window / 10
                }

    return best_result

def optimize_interval_with_loss(
    df, well_name,
    mu_o=0.43, mu_w=0.33,
    P_res=30*1.013e5, P_well=20*1.013e5,
    r_w=0.1, r_e=500, h=0.1, Soi=0.28, B=1.2,
    prise_oil=15000, prise_w=800,
    window_sizes_min_m=4, window_sizes_max_m=35,
    fact_start_depth=None, fact_end_depth=None,
    fact_q=None, fact_wc=None, fact_time=None
):
    """
    Оптимизация интервала испытания и оценка недополученной прибыли.
    """
    well = df[df.well == well_name].reset_index(drop=True)

    # --- расчёт фазовых потоков ---
    well['o'] = (well['fill_cb_res'] * (1 - well['KVS_fin']) - Soi) / mu_o
    well['o'] = np.clip(well['o'], 0, 1)
    o = well['o']
    kv = 1 - well['fill_cb_res'] * (1 - well['KVS_fin'])
    w = (kv - well['KVS_fin']) / mu_w

    well['f_o'] = o / (o + w)
    well['Q_total'] = (2 * np.pi * well["KVS_fin"] * h * (P_res - P_well)) / (B * np.log(r_e / r_w))
    well['Q_oil'] = well['Q_total'] * well['f_o']
    well['Q_wat'] = well['Q_total'] * (1 - well['f_o'])

    # --- 1. Подгонка коэффициента recalc под фактические данные ---
    if fact_start_depth is not None and fact_end_depth is not None and fact_q is not None:
        fact_window = well[(well['DEPTH'] >= fact_start_depth) & (well['DEPTH'] <= fact_end_depth)]
        if len(fact_window) == 0:
            raise ValueError("Фактический интервал не найден в данных")

        model_q = fact_window['Q_oil'].sum() + fact_window['Q_wat'].sum()
        reculc = model_q / fact_q if fact_q > 0 else 1.0
    else:
        reculc = 1.0

    # --- 2. Прибыль по факту ---
    if fact_q is not None and fact_wc is not None and fact_time is not None:
        fact_q_oil = fact_q * (1 - fact_wc)
        fact_q_wat = fact_q * fact_wc
        fact_profit = (prise_oil * fact_q_oil - prise_w * fact_q_wat) * fact_time
    else:
        fact_profit = None

    # --- 3. Поиск оптимального интервала ---
    window_sizes = range(window_sizes_min_m*10, window_sizes_max_m*10)
    best = {"profit": -np.inf}

    for window in window_sizes:
        for start in range(0, len(well) - window + 1):
            end = start + window
            df_window = well.iloc[start:end]

            Q_oil_sum = df_window['Q_oil'].sum() / reculc
            Q_wat_sum = df_window['Q_wat'].sum() / reculc
            profit = (prise_oil * Q_oil_sum - prise_w * Q_wat_sum) * fact_time

            if profit > best["profit"]:
                best = {
                    "profit": profit,
                    "window": window,
                    "start": start,
                    "end": end,
                    "Q_oil_sum": Q_oil_sum,
                    "Q_wat_sum": Q_wat_sum,
                    "start_depth": df_window.iloc[0]['DEPTH'],
                    "end_depth": df_window.iloc[-1]['DEPTH']
                }

    # --- 4. Считаем недополученную прибыль ---
    if fact_profit is not None:
        loss = best["profit"] - fact_profit
    else:
        loss = None

    return {
        "fact_profit": fact_profit,
        "optimal_profit": best["profit"],
        "loss": loss,
        "optimal_interval": (best["start_depth"], best["end_depth"]),
        "optimal_window_m": best["window"]/10,
        "reculc": reculc
    }

# Боковая панель с настройками
st.sidebar.header("⚙️ Параметры анализа")

# Загрузка данных
st.sidebar.subheader("📁 Загрузка данных")
uploaded_file = st.sidebar.file_uploader(
    "Загрузите Excel файл с данными скважин",
    type=['xlsx', 'xls'],
    help="Файл должен содержать колонки: well, DEPTH, fill_cb_res, KVS_fin"
)

# Параметры скважины
st.sidebar.subheader("🛢️ Параметры скважины")
# well_name будет определен после загрузки данных

# Физические параметры
st.sidebar.subheader("🔬 Физические параметры")
mu_o = st.sidebar.slider("Вязкость нефти (Па·с)", 0.1, 1.0, 0.43, 0.01)
mu_w = st.sidebar.slider("Вязкость воды (Па·с)", 0.1, 1.0, 0.33, 0.01)
P_res = st.sidebar.slider("Пластовое давление (атм)", 10, 50, 30, 1) * 1.013e5
P_well = st.sidebar.slider("Давление на устье (атм)", 5, 30, 20, 1) * 1.013e5
r_w = st.sidebar.slider("Радиус скважины (м)", 0.05, 0.2, 0.1, 0.01)
r_e = st.sidebar.slider("Радиус дренирования (м)", 100, 1000, 500, 10)
h = st.sidebar.slider("Толщина пласта (м)", 0.05, 0.5, 0.1, 0.01)
Soi = st.sidebar.slider("Остаточная нефтеёмкость", 0.1, 0.5, 0.28, 0.01)
B = st.sidebar.slider("Объемный коэффициент", 1.0, 2.0, 1.2, 0.1)

# Экономические параметры
st.sidebar.subheader("💰 Экономические параметры")
prise_oil = st.sidebar.number_input("Цена нефти (руб/м³)", 5000, 50000, 15000, 500)
prise_w = st.sidebar.number_input("Утилизация воды (руб/м³)", 100, 2000, 800, 50)
opex_liquid = st.sidebar.number_input("OPEX на жидкость (руб/м³)", 50, 1000, 250, 25)
ndpi_rate = st.sidebar.slider("НДПИ от выручки", 0.0, 0.1, 0.01, 0.001)
reculc = st.sidebar.number_input("Коэффициент пересчёта", 100000, 5000000, 351182, 100000)

# Параметры окна
st.sidebar.subheader("📏 Параметры окна")
window_min = st.sidebar.slider("Минимальная длина окна (м)", 1, 20, 4, 1)
window_max = st.sidebar.slider("Максимальная длина окна (м)", 10, 50, 35, 1)

# Дополнительные параметры для второй функции
st.sidebar.subheader("📊 Фактические данные (для анализа потерь)")
fact_start_depth = st.sidebar.number_input("Фактическая глубина начала (м)", 0, 5000, 2828, 1)
fact_end_depth = st.sidebar.number_input("Фактическая глубина конца (м)", 0, 5000, 2848, 1)
fact_q = st.sidebar.number_input("Фактический дебит (м³/сут)", 0.0, 1000.0, 20.0, 0.1)
fact_wc = st.sidebar.slider("Фактическая обводнённость", 0.0, 1.0, 0.7, 0.01)
fact_time = st.sidebar.number_input("Время работы скважины (сутки)", 1, 3650, 365, 1)

# Основной контент
if uploaded_file is not None:
    try:
        # Загрузка данных
        df = pd.read_excel(uploaded_file)
        
        # Проверка наличия необходимых колонок
        required_columns = ['well', 'DEPTH', 'fill_cb_res', 'KVS_fin']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Отсутствуют необходимые колонки: {missing_columns}")
            st.info("Требуемые колонки: well, DEPTH, fill_cb_res, KVS_fin")
        else:
            # Получение списка доступных скважин
            available_wells = df['well'].unique()
            
            # Выбор скважины из списка
            st.sidebar.subheader("🛢️ Выбор скважины")
            well_name = st.sidebar.selectbox(
                "Выберите скважину для анализа",
                options=available_wells,
                index=0 if len(available_wells) > 0 else None,
                help="Выберите скважину из загруженных данных"
            )
            
            if well_name:
                # Отображение информации о выбранной скважине
                st.success(f"✅ Выбрана скважина: **{well_name}**")
                
                # Показываем статистику по скважине
                well_data = df[df.well == well_name]
                st.info(f"📊 Данные скважины: {len(well_data)} точек, глубина от {well_data['DEPTH'].min():.1f} до {well_data['DEPTH'].max():.1f} м")
                
                # Создание вкладок
                tab1, tab2 = st.tabs(["🔍 Оптимизация интервала", "💰 Анализ потерь"])
                
                with tab1:
                    st.header("🔍 Оптимизация интервала испытания")
                    st.markdown("Поиск оптимального интервала по максимальной прибыли")
                    
                    # Выполнение оптимизации
                    with st.spinner("🔄 Выполняется оптимизация..."):
                        result = optimize_interval(
                            df, well_name,
                            mu_o=mu_o, mu_w=mu_w,
                            P_res=P_res, P_well=P_well,
                            r_w=r_w, r_e=r_e, h=h, Soi=Soi, B=B,
                            prise_oil=prise_oil, prise_w=prise_w,
                            opex_liquid=opex_liquid, ndpi_rate=ndpi_rate,
                            reculc=reculc,
                            window_sizes_min_m=window_min,
                            window_sizes_max_m=window_max
                        )
                    
                    # Отображение результатов
                    st.success("✅ Оптимизация завершена!")
                    
                    # Основные метрики
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Чистая прибыль",
                            f"{result['profit']:,.0f} руб",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "Добыча нефти",
                            f"{result['Q_oil']:.2f} м³",
                            delta=None
                        )
                    
                    with col3:
                        st.metric(
                            "Добыча воды",
                            f"{result['Q_wat']:.2f} м³",
                            delta=None
                        )
                    
                    with col4:
                        st.metric(
                            "Длина интервала",
                            f"{result['window_m']:.1f} м",
                            delta=None
                        )
                    
                    # Детальная информация
                    st.subheader("📊 Детальные результаты")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Экономические показатели:**")
                        st.write(f"• Выручка: {result['revenue']:,.2f} руб")
                        st.write(f"• НДПИ: {result['ndpi']:,.2f} руб")
                        st.write(f"• Затраты: {result['costs']:,.2f} руб")
                        st.write(f"• Чистая прибыль: {result['profit']:,.2f} руб")
                    
                    with col2:
                        st.markdown("**Технические показатели:**")
                        st.write(f"• Глубина начала: {result['depth_start']:.1f} м")
                        st.write(f"• Глубина конца: {result['depth_end']:.1f} м")
                        st.write(f"• Длина интервала: {result['window_m']:.1f} м")
                        st.write(f"• Добыча нефти: {result['Q_oil']:.2f} м³")
                        st.write(f"• Добыча воды: {result['Q_wat']:.2f} м³")
                    
                    # Визуализация
                    st.subheader("📈 Визуализация данных")
                    
                    # График профиля скважины
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Профиль скважины', 'Фазовые потоки'),
                        specs=[[{"secondary_y": True}, {"secondary_y": False}]]
                    )
                    
                    # Профиль скважины
                    fig.add_trace(
                        go.Scatter(
                            x=well_data['fill_cb_res'],
                            y=well_data['DEPTH'],
                            mode='lines',
                            name='Заполненность',
                            line=dict(color='blue', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=well_data['KVS_fin'],
                            y=well_data['DEPTH'],
                            mode='lines',
                            name='КВС',
                            line=dict(color='red', width=2),
                            yaxis='y2'
                        ),
                        row=1, col=1, secondary_y=True
                    )
                    
                    # Фазовые потоки
                    well_data = well_data.copy()  # Создаем копию для безопасного редактирования
                    well_data.loc[:, 'o'] = (well_data['fill_cb_res'] * (1 - well_data['KVS_fin']) - Soi) / mu_o
                    well_data.loc[:, 'o'] = np.clip(well_data['o'], 0, 1)
                    o = well_data['o']
                    kv = 1 - well_data['fill_cb_res'] * (1 - well_data['KVS_fin'])
                    w = (kv - well_data['KVS_fin']) / mu_w
                    well_data.loc[:, 'f_o'] = o / (o + w)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=well_data['f_o'],
                            y=well_data['DEPTH'],
                            mode='lines',
                            name='Доля нефти',
                            line=dict(color='green', width=2)
                        ),
                        row=1, col=2
                    )
                    
                    # Выделение оптимального интервала с заливкой
                    optimal_start = result['depth_start']
                    optimal_end = result['depth_end']
                    
                    # Заливка для оптимального интервала на первом графике
                    fig.add_hrect(
                        y0=optimal_start, y1=optimal_end,
                        fillcolor="yellow", opacity=0.4,
                        layer="below", line_width=2,
                        line_color="orange",
                        annotation_text="Оптимальный интервал",
                        annotation_position="top left",
                        row=1, col=1
                    )
                    
                    # Заливка для оптимального интервала на втором графике
                    fig.add_hrect(
                        y0=optimal_start, y1=optimal_end,
                        fillcolor="yellow", opacity=0.4,
                        layer="below", line_width=2,
                        line_color="orange",
                        annotation_text="Оптимальный интервал",
                        annotation_position="top left",
                        row=1, col=2
                    )
                    
                    fig.update_layout(
                        height=800,
                        title_text="Анализ скважины",
                        showlegend=True
                    )
                    
                    # Обратная шкала глубины (глубина увеличивается вниз)
                    fig.update_xaxes(title_text="Заполненность", row=1, col=1)
                    fig.update_yaxes(
                        title_text="Глубина (м)", 
                        row=1, col=1, 
                        autorange="reversed",
                        dtick=50,  # деления каждые 50 метров
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="lightgray"
                    )
                    fig.update_xaxes(title_text="Доля нефти", row=1, col=2)
                    fig.update_yaxes(
                        title_text="Глубина (м)", 
                        row=1, col=2, 
                        autorange="reversed",
                        dtick=50,  # деления каждые 50 метров
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="lightgray"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    st.header("💰 Анализ потерь и недополученной прибыли")
                    st.markdown("Сравнение фактических результатов с оптимальными")
                    
                    # Выполнение анализа потерь
                    with st.spinner("🔄 Выполняется анализ потерь..."):
                        result_loss = optimize_interval_with_loss(
                            df, well_name,
                            mu_o=mu_o, mu_w=mu_w,
                            P_res=P_res, P_well=P_well,
                            r_w=r_w, r_e=r_e, h=h, Soi=Soi, B=B,
                            prise_oil=prise_oil, prise_w=prise_w,
                            window_sizes_min_m=window_min, window_sizes_max_m=window_max,
                            fact_start_depth=fact_start_depth, fact_end_depth=fact_end_depth,
                            fact_q=fact_q, fact_wc=fact_wc, fact_time=fact_time
                        )
                    
                    # Отображение результатов
                    st.success("✅ Анализ потерь завершен!")
                    
                    # Основные метрики
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Фактическая прибыль",
                            f"{result_loss['fact_profit']:,.0f} руб" if result_loss['fact_profit'] else "N/A",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "Оптимальная прибыль",
                            f"{result_loss['optimal_profit']:,.0f} руб",
                            delta=None
                        )
                    
                    with col3:
                        st.metric(
                            "Недополученная прибыль",
                            f"{result_loss['loss']:,.0f} руб" if result_loss['loss'] else "N/A",
                            delta=f"-{result_loss['loss']:,.0f} руб" if result_loss['loss'] and result_loss['loss'] > 0 else None
                        )
                    
                    with col4:
                        st.metric(
                            "Коэффициент пересчёта",
                            f"{result_loss['reculc']:.3f}",
                            delta=None
                        )
                    
                    # Детальная информация
                    st.subheader("📊 Детальный анализ")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Фактические данные:**")
                        st.write(f"• Интервал: {fact_start_depth}–{fact_end_depth} м")
                        st.write(f"• Дебит: {fact_q} м³/сут")
                        st.write(f"• Обводнённость: {fact_wc:.1%}")
                        st.write(f"• Время работы: {fact_time} суток")
                        st.write(f"• Прибыль: {result_loss['fact_profit']:,.0f} руб" if result_loss['fact_profit'] else "• Прибыль: N/A")
                    
                    with col2:
                        st.markdown("**Оптимальные данные:**")
                        st.write(f"• Интервал: {result_loss['optimal_interval'][0]:.1f}–{result_loss['optimal_interval'][1]:.1f} м")
                        st.write(f"• Длина окна: {result_loss['optimal_window_m']:.1f} м")
                        st.write(f"• Прибыль: {result_loss['optimal_profit']:,.0f} руб")
                        st.write(f"• Потери: {result_loss['loss']:,.0f} руб" if result_loss['loss'] else "• Потери: N/A")
                        st.write(f"• Коэффициент: {result_loss['reculc']:.3f}")
                    
                    # Визуализация сравнения
                    st.subheader("📈 Сравнительный анализ")
                    
                    # График сравнения интервалов
                    fig_comp = go.Figure()
                    
                    # Фактический интервал
                    fig_comp.add_vrect(
                        x0=0, x1=1,
                        y0=fact_start_depth, y1=fact_end_depth,
                        fillcolor="red", opacity=0.3,
                        layer="below", line_width=2,
                        annotation_text="Фактический интервал",
                        annotation_position="top right"
                    )
                    
                    # Оптимальный интервал
                    fig_comp.add_vrect(
                        x0=0, x1=1,
                        y0=result_loss['optimal_interval'][0], y1=result_loss['optimal_interval'][1],
                        fillcolor="green", opacity=0.3,
                        layer="below", line_width=2,
                        annotation_text="Оптимальный интервал",
                        annotation_position="top left"
                    )
                    
                    # Профиль скважины
                    fig_comp.add_trace(
                        go.Scatter(
                            x=well_data['fill_cb_res'],
                            y=well_data['DEPTH'],
                            mode='lines',
                            name='Заполненность',
                            line=dict(color='blue', width=2)
                        )
                    )
                    
                    fig_comp.update_layout(
                        height=800,
                        title_text="Сравнение фактического и оптимального интервалов",
                        xaxis_title="Заполненность",
                        yaxis_title="Глубина (м)",
                        showlegend=True
                    )
                    
                    # Обратная шкала глубины с детализацией
                    fig_comp.update_yaxes(
                        autorange="reversed",
                        dtick=50,  # деления каждые 50 метров
                        showgrid=True,
                        gridwidth=1,
                        gridcolor="lightgray"
                    )
                    
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Экономический анализ
                    if result_loss['fact_profit'] and result_loss['loss']:
                        st.subheader("💰 Экономический анализ потерь")
                        
                        loss_percentage = (result_loss['loss'] / result_loss['fact_profit']) * 100 if result_loss['fact_profit'] > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Процент потерь",
                                f"{loss_percentage:.1f}%",
                                delta=f"-{loss_percentage:.1f}%"
                            )
                        
                        with col2:
                            st.metric(
                                "Потенциал роста",
                                f"{((result_loss['optimal_profit'] / result_loss['fact_profit']) - 1) * 100:.1f}%",
                                delta=f"+{((result_loss['optimal_profit'] / result_loss['fact_profit']) - 1) * 100:.1f}%"
                            )
                        
                        with col3:
                            st.metric(
                                "Эффективность",
                                f"{(result_loss['fact_profit'] / result_loss['optimal_profit']) * 100:.1f}%",
                                delta=f"-{(1 - result_loss['fact_profit'] / result_loss['optimal_profit']) * 100:.1f}%"
                            )
                
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке файла: {str(e)}")

else:
    # Инструкции для пользователя
    st.info("👆 Пожалуйста, загрузите Excel файл с данными скважин в боковой панели")
    
    # Показываем информацию о том, что нужно загрузить файл для выбора скважины
    st.sidebar.info("📁 Загрузите файл для выбора скважины")
    
    st.markdown("""
    ### 📋 Требования к данным
    
    Файл должен содержать следующие колонки:
    - `well` - название скважины
    - `DEPTH` - глубина (м)
    - `fill_cb_res` - заполненность коллектора
    - `KVS_fin` - коэффициент вытеснения нефти водой
    
    ### 🔧 Возможности приложения
    
    - **Оптимизация интервалов** - поиск наиболее прибыльного интервала испытания
    - **💰 Анализ потерь** - сравнение фактических и оптимальных результатов
    - **Интерактивные параметры** - настройка физических и экономических параметров
    - **Визуализация** - графики профиля скважины и фазовых потоков
    - **Экономический анализ** - расчет прибыли, затрат и налогов
    
    ### 🛢️ Выбор скважины
    
    После загрузки файла в боковой панели появится выпадающий список со всеми доступными скважинами из ваших данных.
    """)

# Футер
st.markdown("---")
st.markdown("🛢️ **Приложение для оптимизации интервалов испытания нефтяных скважин**")
