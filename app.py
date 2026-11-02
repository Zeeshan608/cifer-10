import numpy as np, streamlit as st
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
EMOJI = ["✈️", "🚗", "🐦", "🐱", "🦌", "🐶", "🐸", "🐴", "🚢", "🚚"]

st.set_page_config(page_title="CIFAR-10 Vision AI", page_icon="🔮", layout="centered")

# Custom CSS: gradient background, glass cards, gradient title and probability bars
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);}
.stApp, .stApp p, .stApp label, .stApp span {color:#ffffff;}
.hero {text-align:center; padding:1rem 0 0.5rem;}
.hero h1 {font-size:3rem; font-weight:800; background:linear-gradient(90deg,#00dbde,#fc00ff);
          -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.hero p {color:#b9b6d3;}
.card {background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.18);
       border-radius:20px; padding:1.5rem; text-align:center; margin-top:1rem;}
.big {font-size:4.5rem; line-height:1;}
.label {font-size:2rem; font-weight:700; text-transform:capitalize;}
.conf {color:#00dbde; font-size:1.2rem;}
.bar-bg {background:rgba(255,255,255,0.12); border-radius:10px; height:14px; margin:4px 0 14px;}
.bar {height:14px; border-radius:10px; background:linear-gradient(90deg,#00dbde,#fc00ff);}
.row {display:flex; justify-content:space-between; text-align:left; text-transform:capitalize;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🔮 Vision AI</h1>'
            '<p>Upload a photo and a fine-tuned ResNet50 will tell you what it sees.<br>'
            'It knows 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.</p></div>',
            unsafe_allow_html=True)

@st.cache_resource  # load the model once and reuse it for every user
def load_model():
    return keras.models.load_model("cifar10_resnet50.keras")

def predict(img):
    arr = np.array(img.convert("RGB").resize((32, 32)), dtype="float32")[None]  # same steps as training
    return load_model().predict(preprocess_input(arr), verbose=0)[0]

file = st.file_uploader("Drop an image here", type=["png", "jpg", "jpeg", "webp"])

if file:
    img = Image.open(file)
    left, right = st.columns(2)
    left.image(img, use_container_width=True)
    with st.spinner("Analyzing..."):
        probs = predict(img)
    top = probs.argsort()[::-1][:3]
    right.markdown(f'<div class="card"><div class="big">{EMOJI[top[0]]}</div>'
                   f'<div class="label">{CLASSES[top[0]]}</div>'
                   f'<div class="conf">{probs[top[0]]:.1%} confident</div></div>', unsafe_allow_html=True)
    st.markdown("### Top 3 predictions")
    for i in top:
        st.markdown(f'<div class="row"><span>{EMOJI[i]} {CLASSES[i]}</span><span>{probs[i]:.1%}</span></div>'
                    f'<div class="bar-bg"><div class="bar" style="width:{probs[i]*100:.1f}%"></div></div>',
                    unsafe_allow_html=True)
else:
    st.info("👆 Upload a clear photo with one main object for the best results.")
