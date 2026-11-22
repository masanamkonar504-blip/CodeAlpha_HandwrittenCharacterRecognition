
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# ----------------------------------
# PAGE SETTINGS
# ----------------------------------

st.set_page_config(
    page_title="NeuroWrite AI",
    page_icon="✍️",
    layout="wide"
)

# ----------------------------------
# CUSTOM DESIGN
# ----------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #0b1020,
        #21133d,
        #101c35
    );
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

div.stButton > button {
    background: linear-gradient(
        90deg,
        #8b5cf6,
        #06b6d4
    );
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# LOAD TRAINED MODEL
# ----------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "neurowrite_model.keras"
    )

model = load_model()

# ----------------------------------
# WEBSITE HEADING
# ----------------------------------

st.title("✍️ NeuroWrite AI")

st.subheader(
    "Intelligent Handwritten Character Recognition"
)

st.write(
    "Draw any uppercase English letter (A-Z) "
    "or digit (0-9) below."
)

st.markdown("---")

# ----------------------------------
# MAIN LAYOUT
# ----------------------------------

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 🎨 Drawing Canvas")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=15,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=280,
        height=280,
        drawing_mode="freedraw",
        return_image_data=True,
        key="canvas"
    )

    predict_button = st.button(
        "✨ Predict Character",
        use_container_width=True
    )

with col2:

    st.markdown("### 🧠 AI Prediction")

    if predict_button:

        if (
            canvas_result is not None
            and canvas_result.image_data is not None
        ):

            # Convert canvas to grayscale
            image = Image.fromarray(
                canvas_result.image_data.astype("uint8")
            ).convert("L")

            # Invert black writing to white
            image = ImageOps.invert(image)

            # Find character boundaries
            image_array = np.array(image)

            coords = np.argwhere(image_array > 30)

            if len(coords) > 0:

                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                # Crop character
                image = image.crop((
                    x_min,
                    y_min,
                    x_max + 1,
                    y_max + 1
                ))

                # Resize while preserving proportions
                image.thumbnail((20, 20))

                # Center in 28x28 image
                final_image = Image.new(
                    "L",
                    (28, 28),
                    0
                )

                x_offset = (28 - image.width) // 2
                y_offset = (28 - image.height) // 2

                final_image.paste(
                    image,
                    (x_offset, y_offset)
                )

                # Prepare CNN input
                image_array = (
                    np.array(final_image)
                    .astype("float32") / 255.0
                )

                image_array = image_array.reshape(
                    1, 28, 28, 1
                )

                # AI prediction
                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                top_indices = np.argsort(
                    predictions
                )[-3:][::-1]

                best_index = int(top_indices[0])

                # Convert prediction index to character
                if best_index < 10:
                    character = str(best_index)
                else:
                    character = chr(
                        ord("A") + best_index - 10
                    )

                # Display main prediction
                st.markdown(
                    f"""
                    <h1 style="
                    font-size:90px;
                    text-align:center;
                    color:#67e8f9 !important;
                    ">
                    {character}
                    </h1>
                    """,
                    unsafe_allow_html=True
                )

                # Confidence
                confidence = float(
                    predictions[best_index] * 100
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                # Top 3 predictions
                st.markdown("### 🏆 Top 3 Predictions")

                for index in top_indices:

                    index = int(index)

                    if index < 10:
                        label = str(index)
                    else:
                        label = chr(
                            ord("A") + index - 10
                        )

                    score = float(
                        predictions[index] * 100
                    )

                    st.write(
                        f"**{label}** — {score:.2f}%"
                    )

                    st.progress(score / 100)

            else:
                st.warning(
                    "Please draw a character first!"
                )

        else:
            st.warning(
                "Please draw a character first!"
            )

# ----------------------------------
# FOOTER
# ----------------------------------

st.markdown("---")

st.caption(
    "NeuroWrite AI | Powered by CNN and TensorFlow"
)
