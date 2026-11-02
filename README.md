# CIFAR-10 Vision AI

A Streamlit app that classifies an uploaded image into one of the ten CIFAR-10 classes using a ResNet50 model.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The model file, `cifar10_resnet50.keras`, is stored with Git LFS.

## Deploy on Streamlit Community Cloud

1. Select the `Zeeshan608/cifer-10` repository and the `main` branch.
2. Set the app file to `app.py`.
3. Choose Python 3.12 in the advanced settings.
4. Deploy the app.
