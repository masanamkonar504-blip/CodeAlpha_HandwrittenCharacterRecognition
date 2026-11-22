
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Page settings
st.set_page_config(
    page_title="NeuroWrite AI",
    page_icon="✍️",
    layout="wide"
)

# Custom design
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #21133d, #101c35);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
div.stButton > button {
    background: linear-gradient(90deg, #8b5cf6, #06b6d4);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("neurowrite_model.keras")

model = load_model()

# Website heading
st.title("✍️ NeuroWrite AI")
st.subheader("Intelligent Handwritten Character Recognition")
st.write("Draw any uppercase letter (A-Z) or digit (0-9) below.")

st.markdown("---")

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
        key="canvas"
    )

    predict_button = st.button(
        "✨ Predict Character",
        use_container_width=True
    )

with col2:
    st.markdown("### 🧠 AI Prediction")

    if predict_button:
        if canvas_result.json_data is not None and len(canvas_result.json_data.get("objects", [])) > 0:

            image = Image.fromarray(
                canvas_result.image_data.astype("uint8")
            ).convert("L")
            image = ImageOps.invert(image)

            image = image.resize((28, 28))

            image_array = np.array(image).astype("float32") / 255.0
            image_array = image_array.reshape(1, 28, 28, 1)

            predictions = model.predict(image_array, verbose=0)[0]

            top_indices = np.argsort(predictions)[-3:][::-1]

            best_index = top_indices[0]

            if best_index < 10:
                character = str(best_index)
            else:
                character = chr(ord("A") + best_index - 10)

            st.markdown(
                f"<h1 style='font-size:90px;text-align:center;color:#67e8f9 !important;'>{character}</h1>",
                unsafe_allow_html=True
            )

            st.metric(
                "Confidence",
                f"{predictions[best_index] * 100:.2f}%"
            )

            st.markdown("### 🏆 Top 3 Predictions")

            for index in top_indices:
                if index < 10:
                    label = str(index)
                else:
                    label = chr(ord("A") + index - 10)

                confidence = predictions[index] * 100

                st.write(f"**{label}** — {confidence:.2f}%")
                st.progress(float(confidence / 100))

        else:
            st.warning("Please draw a character first!")

st.markdown("---")
st.caption("NeuroWrite AI | Powered by CNN and TensorFlow")
