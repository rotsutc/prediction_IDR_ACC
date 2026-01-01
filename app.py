import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
import joblib

st.set_page_config(
    page_title="Prediction of Story Drift and Peak Floor Acceleration",
    page_icon="icon/avatars 500x500.jpg",          # có thể dùng emoji hoặc đường dẫn ảnh
    layout="wide",           # "centered" hoặc "wide"
    initial_sidebar_state="expanded",
)

# ================================
# Parameters of the steel building
# ================================

n_floors = 10
n_frames_x = 4
n_frames_y = 3
bay_length = 6.0     # meters
h1 = 4.5              # 1st floor height
h_other = 3.6         # height of each floor above

# Compute floor elevations
z_levels = [0]
z_levels.append(h1)
for _ in range(n_floors - 1):
    z_levels.append(z_levels[-1] + h_other)

# Grid coordinates
x_pos = np.arange(n_frames_x) * bay_length
y_pos = np.arange(n_frames_y) * bay_length


# ========================================
# Function to create 3D frame structure
# ========================================

def draw_3d_frame():
    fig = go.Figure()

    # Draw columns
    for x in x_pos:
        for y in y_pos:
            for f in range(n_floors):
                fig.add_trace(go.Scatter3d(
                    x=[x, x],
                    y=[y, y],
                    z=[z_levels[f], z_levels[f+1]],
                    mode="lines",
                    line=dict(color="black", width=5)
                ))

    # Draw beams (X direction)
    for f in range(1, n_floors + 1):
        for y in y_pos:
            for i in range(len(x_pos) - 1):
                fig.add_trace(go.Scatter3d(
                    x=[x_pos[i], x_pos[i+1]],
                    y=[y, y],
                    z=[z_levels[f], z_levels[f]],
                    mode="lines",
                    line=dict(color="blue", width=4)
                ))

    # Draw beams (Y direction)
    for f in range(1, n_floors + 1):
        for x in x_pos:
            for j in range(len(y_pos) - 1):
                fig.add_trace(go.Scatter3d(
                    x=[x, x],
                    y=[y_pos[j], y_pos[j+1]],
                    z=[z_levels[f], z_levels[f]],
                    mode="lines",
                    line=dict(color="red", width=4)
                ))

    fig.update_layout(
        width=650,
        height=700,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=False
    )
    return fig


# ================================================
# Draw side-view projection
# ================================================

def draw_projection(direction="X"):
    fig = go.Figure()

    if direction == "X":
        # Collapse X → draw Y-Z frame
        for y in y_pos:
            for f in range(n_floors):
                fig.add_trace(go.Scatter(
                    x=[y, y],
                    y=[z_levels[f], z_levels[f+1]],
                    mode="lines",
                    line=dict(color="black", width=3)
                ))
            for f in range(1, n_floors + 1):
                for j in range(len(y_pos) - 1):
                    fig.add_trace(go.Scatter(
                        x=[y_pos[j], y_pos[j+1]],
                        y=[z_levels[f], z_levels[f]],
                        mode="lines",
                        line=dict(color="blue", width=2)
                    ))

        xlabel = "Y (m)"

    else:  # direction == "Y"
        for x in x_pos:
            for f in range(n_floors):
                fig.add_trace(go.Scatter(
                    x=[x, x],
                    y=[z_levels[f], z_levels[f+1]],
                    mode="lines",
                    line=dict(color="black", width=3)
                ))
            for f in range(1, n_floors + 1):
                for i in range(len(x_pos) - 1):
                    fig.add_trace(go.Scatter(
                        x=[x_pos[i], x_pos[i+1]],
                        y=[z_levels[f], z_levels[f]],
                        mode="lines",
                        line=dict(color="red", width=2)
                    ))

        xlabel = "X (m)"

    fig.update_layout(
        width=650,
        height=700,
        xaxis_title=xlabel,
        yaxis_title="Z (m)",
        yaxis=dict(scaleanchor="x", scaleratio=0.5),
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=False
    )
    return fig


# ================================
# Streamlit UI
# ================================

st.markdown(
    r"""
    <h1 style="text-align:center;">
    Prediction of Story Drift and Peak Floor Acceleration of a 10-Story Steel Building
    with 4 Frames Along X Direction and 3 Frames Along Y Direction
    </h1>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("3D Steel Building Model")
    st.plotly_chart(draw_3d_frame(), width='stretch')

with col2:
    st.subheader("Side Projection View")

    direction = st.selectbox(
        "Choose direction for projection:",
        ["X", "Y"],
        index=0
    )

    st.plotly_chart(draw_projection(direction), width='stretch')


# ================================================================
# Read AT2 file
# ================================================================

def read_AT2_file(filepath, show_info=True):
    with open(filepath, "r") as f:
        lines = f.readlines()

    source_name = lines[0].strip()
    location_date = lines[1].strip()
    unit_info = lines[2].strip()

    line4 = lines[3].strip()

    m_npts = re.search(r"NPTS\s*=\s*(\d+)", line4, re.IGNORECASE)
    if not m_npts:
        raise ValueError("Cannot find NPTS")
    NPTS = int(m_npts.group(1))

    m_dt = re.search(
        r"DT\s*=\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)",
        line4,
        re.IGNORECASE
    )
    if not m_dt:
        raise ValueError("Cannot find DT")
    DT = float(m_dt.group(1))

    accel_values = []
    for line in lines[4:]:
        accel_values += [float(x) for x in line.split()]

    if len(accel_values) == 0 or accel_values[0] != 0:
        accel_values.insert(0, 0.0)
        NPTS += 1

    time_history = [i * DT for i in range(NPTS)]

    return {
        "source_name": source_name,
        "location_date": location_date,
        "unit_info": unit_info,
        "NPTS": NPTS,
        "DT": DT,
        "time_history": time_history,
        "acceleration_history": accel_values
    }


# ================================================================
# Extract 1000 points
# ================================================================

def get_n_points(at2_data, Dir, n_points=1000):
    try:
        acc = np.array(at2_data["acceleration_history"])
        L = len(acc)

        # ==== Sliding window chọn 1000 điểm có tổng |acc| lớn nhất ====
        abs_acc = np.abs(acc)

        if L >= n_points:
            cumsum = np.cumsum(np.insert(abs_acc, 0, 0))
            window_sum = cumsum[n_points:] - cumsum[:-n_points]
            idx = np.argmax(window_sum)

            best_segment = acc[idx : idx + n_points]  # lấy acc gốc
        else:
            best_segment = np.concatenate([acc, np.zeros(n_points - L)])
            idx = 0

        # ---- Xuất một dòng dữ liệu ----
        out_row = {
            "Direction": Dir,
            "DT": at2_data["DT"],
            "Start_Index": idx,
            "End_Index": idx + n_points - 1
        }

        for i in range(n_points):
            out_row[f"Acc_{i+1}"] = best_segment[i]

        df_out = pd.DataFrame([out_row])
        return df_out

    except Exception as e:
        print(f"❌ Lỗi khi xử lý AT2 data: {e}")
        return None


# ================================================================
# UI for AT2
# ================================================================

st.write("---")
st.subheader("Load AT2 Ground Motion File")

uploaded_at2 = st.file_uploader(
    "Open file AT2",
    type=["AT2"],
    accept_multiple_files=False
)

if uploaded_at2 is not None:
    st.success(f"Selected file: **{uploaded_at2.name}**")

    temp_path = f"./{uploaded_at2.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_at2.getbuffer())

    at2_data = read_AT2_file(temp_path, show_info=False)

    st.write("### File Information")
    st.write(f"**Source Name:** {at2_data['source_name']}")
    st.write(f"**Location & Date:** {at2_data['location_date']}")
    st.write(f"**Unit Info:** {at2_data['unit_info']}")
    st.write(f"**Number of Points (NPTS):** {at2_data['NPTS']}")

    t = at2_data["time_history"]
    acc = at2_data["acceleration_history"]

    fig_at2 = go.Figure()
    fig_at2.add_trace(go.Scatter(
        x=t,
        y=acc,
        mode="lines",
        line=dict(width=2)
    ))

    fig_at2.update_layout(
        width=900,
        height=350,
        title="Ground Motion Time History",
        xaxis_title="Time (s)",
        yaxis_title="Acceleration (g)"
    )

    st.plotly_chart(fig_at2, width='stretch')

    # Mapping X → 1, Y → 2
    Dir = 1 if direction == "X" else 2
    st.write(f"**Selected Dir value:** {Dir}")

    df_points = get_n_points(at2_data, Dir, n_points=1000)

    # ====================================
    # Plot AT2 + Highlight 1000-point segment
    # ====================================
    st.write("### Highlighted 1000-Point Segment on AT2 Record")

    fig_highlight = go.Figure()

    # --- Full AT2 (blue) ---
    fig_highlight.add_trace(go.Scatter(
        x=t,
        y=acc,
        mode="lines",
        line=dict(width=1.5, color="blue"),
        name="Full AT2"
    ))

    # --- Determine start index of the 1000-point segment (|acc| sum) ---
    abs_acc = np.abs(np.array(acc))

    n_points = 1000
    L = len(abs_acc)

    if L >= n_points:
        cumsum = np.cumsum(np.insert(abs_acc, 0, 0))
        window_sum = cumsum[n_points:] - cumsum[:-n_points]
        idx = np.argmax(window_sum)
    else:
        idx = 0

    # Create segment time vector based on original time axis
    t_segment = t[idx : idx + n_points]
    segment = acc[idx : idx + n_points]

    # --- Highlight segment (red) ---
    fig_highlight.add_trace(go.Scatter(
        x=t_segment,
        y=segment,
        mode="lines",
        line=dict(width=3, color="red"),
        name="Selected 1000-Point |Acc| Segment"
    ))

    fig_highlight.update_layout(
        width=900,
        height=400,
        title="1000-Point Segment with Maximum Absolute Acceleration Sum",
        xaxis_title="Time (s)",
        yaxis_title="Acceleration (g)"
    )

    st.plotly_chart(fig_highlight, width='stretch')

    model = joblib.load("catboost_multi_gpu.pkl")
    st.success("✅ Đã load mô hình CatBoost!")
    # print("")


    # ===========================================================
    #   NÚT PREDICTION – đặt giữa màn hình
    # ===========================================================


    center_col = st.columns([1, 1, 1])
    pred_btn = center_col[1].button("Prediction")

    PLOT_HEIGHT = 700

    if pred_btn:

        X_cols = ["Direction", "DT"] + [f"Acc_{i}" for i in range(1, 1001)]
        X_input = df_points[X_cols]

        y_pred_input = model.predict(X_input)

        # print(y_pred_input)

        # 10 giá trị IDR + 10 giá trị ACC
        IDR = y_pred_input[0, :10]
        ACC = y_pred_input[0, 10:]

        IDR = np.insert(IDR, 0, 0.0)
        ACC = np.insert(ACC, 0, 0.0)

        st.success("✅ Prediction completed!")
        # =======================================================
        # HIỂN THỊ 3 BIỂU ĐỒ THEO PHƯƠNG NGANG
        # =======================================================
        col_left, col_mid, col_right = st.columns(3)

        # ====== LEFT: hình chiếu cạnh ======
        with col_left:
            fig_proj = draw_projection(direction)
            fig_proj.update_layout(height=PLOT_HEIGHT)
            st.plotly_chart(fig_proj, width='stretch', key="projection_plot")

        # ====== MID: biểu đồ IDR ======
        with col_mid:
            floors = np.arange(0, 11)

            fig_idr = go.Figure()
            fig_idr.add_trace(go.Scatter(
                x=IDR,
                y=floors,
                mode="lines+markers",
                line=dict(width=3),
                marker=dict(size=8)
            ))

            fig_idr.update_layout(
                height=PLOT_HEIGHT,
                title="Inter-Story Drift Ratio (IDR)",
                xaxis_title="IDR [%]",
                yaxis_title="Floor",
                yaxis=dict(range=[0, 10]),
                margin=dict(l=0, r=0, t=40, b=0)
            )

            # Bỏ scaleanchor để trục X không bị lệch
            fig_idr.update_yaxes(scaleanchor=None)


            st.plotly_chart(fig_idr, width='stretch')

        # ====== RIGHT: biểu đồ ACC ======
        with col_right:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=ACC,
                y=floors,
                mode="lines+markers",
                line=dict(width=3),
                marker=dict(size=8)
            ))

            fig_acc.update_layout(
                height=PLOT_HEIGHT,
                title="Peak Floor Acceleration (ACC)",
                xaxis_title="ACC [m/s²]",
                yaxis_title="Floor",
                yaxis=dict(range=[0, 10]),
                margin=dict(l=0, r=0, t=40, b=0)
            )

            fig_acc.update_yaxes(scaleanchor=None)

            st.plotly_chart(fig_acc, width='stretch')

        st.write("Results:")
        # st.dataframe(pd.DataFrame(IDR).T)
        # st.dataframe(pd.DataFrame(ACC).T)
        n_idr = len(IDR)

        df_idr = pd.DataFrame(
            [
                ["IDR"] + list(range(0, n_idr)),
                ["value"] + list(IDR)
            ]
        )

        st.dataframe(df_idr, hide_index=True)

        n_acc = len(ACC)

        df_acc = pd.DataFrame(
            [
                ["ACC"] + list(range(0, n_acc)),
                ["value"] + list(ACC)
            ]
        )

        st.dataframe(df_acc, hide_index=True)
else:
    st.info("No file selected yet.")


