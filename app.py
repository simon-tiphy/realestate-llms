# from flask import Flask, request, jsonify
# from flask_restful import Resource, Api
# from flask_cors import CORS
# from transformers import (
#     pipeline,
#     AutoImageProcessor,
#     AutoModelForImageClassification
# )
# import os
# import torch

# app = Flask(__name__)
# CORS(app) 
# api = Api(app)

# # — your HF repo ID & token —
# HF_MODEL_ID = "andupets/real-estate-image-classification"
# HF_TOKEN    = os.environ.get("HUGGINGFACE_TOKEN", None)

# # — force CPU (or use GPU if available) —
# device = 0 if torch.cuda.is_available() else -1
# app.logger.info(f"Device set to use {'cuda' if device==0 else 'cpu'}")

# # — 1. Load the fast image processor & model with authentication —
# processor = AutoImageProcessor.from_pretrained(
#     HF_MODEL_ID,
#     use_fast=True,
#     use_auth_token=HF_TOKEN
# )
# model = AutoModelForImageClassification.from_pretrained(
#     HF_MODEL_ID,
#     use_auth_token=HF_TOKEN
# )

# # — 2. Build the pipeline (no use_auth_token here) —
# classifier = pipeline(
#     task="image-classification",
#     model=model,
#     feature_extractor=processor,
#     device=device,
#     top_k=5        # return top 5 predictions by default
# )

# # — 3. Health‑check endpoint —
# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify(status="ok"), 200

# # — 4. RESTful resource for image classification —
# class Classify(Resource):
#     def post(self):
#         """
#         POST /api/classify
#         JSON body: { "image_url": "<url_or_path_to_image>" }
#         Returns: { "predictions": [ { "label": str, "score": float }, … ] }
#         """
#         data = request.get_json(force=True) or {}
#         image_ref = data.get("image_url") or data.get("image_path")
#         if not image_ref:
#             return {"error": "No image URL or path provided"}, 400

#         try:
#             results = classifier(image_ref)
#             return {"predictions": results}, 200

#         except Exception as e:
#             app.logger.error(f"Classification failed: {e}")
#             return {"error": "Image classification failed"}, 500

# # — 5. Mount the resource —
# api.add_resource(Classify, "/api/classify")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000, debug=True)


# from flask import Flask, request, jsonify
# from flask_restful import Resource, Api
# from flask_cors import CORS
# from transformers import (
#     pipeline,
#     AutoImageProcessor,
#     AutoModelForImageClassification,
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM
# )
# import os
# import torch

# app = Flask(__name__)
# CORS(app)
# api = Api(app)

# # === MODEL CONFIGS ===
# IMG_MODEL_ID = "andupets/real-estate-image-classification"
# TEXT_MODEL_ID = "mohsenfayyaz/mt5-small-query_realestate_cars-finetuned"
# HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)
# device = 0 if torch.cuda.is_available() else -1

# app.logger.info(f"Device set to use {'cuda' if device==0 else 'cpu'}")

# # === IMAGE CLASSIFICATION SETUP ===
# img_processor = AutoImageProcessor.from_pretrained(
#     IMG_MODEL_ID,
#     use_fast=True,
#     use_auth_token=HF_TOKEN
# )
# img_model = AutoModelForImageClassification.from_pretrained(
#     IMG_MODEL_ID,
#     use_auth_token=HF_TOKEN
# )
# classifier = pipeline(
#     task="image-classification",
#     model=img_model,
#     feature_extractor=img_processor,
#     device=device,
#     top_k=5
# )

# # === TEXT2TEXT GENERATION SETUP ===
# text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID, use_auth_token=HF_TOKEN)
# text_model = AutoModelForSeq2SeqLM.from_pretrained(TEXT_MODEL_ID, use_auth_token=HF_TOKEN)
# generator = pipeline(
#     task="text2text-generation",
#     model=text_model,
#     tokenizer=text_tokenizer,
#     device=device
# )

# # === HEALTH CHECK ===
# @app.route("/health", methods=["GET"])
# def health():
#     return jsonify(status="ok"), 200

# # === IMAGE CLASSIFICATION ENDPOINT ===
# class Classify(Resource):
#     def post(self):
#         data = request.get_json(force=True) or {}
#         image_ref = data.get("image_url") or data.get("image_path")
#         if not image_ref:
#             return {"error": "No image URL or path provided"}, 400

#         try:
#             results = classifier(image_ref)
#             return {"predictions": results}, 200
#         except Exception as e:
#             app.logger.error(f"Image classification failed: {e}")
#             return {"error": "Image classification failed"}, 500

# # === TEXT GENERATION ENDPOINT ===
# class Generate(Resource):
#     def post(self):
#         data = request.get_json(force=True) or {}
#         prompt = data.get("prompt", "").strip()
#         if not prompt:
#             return {"error": "No prompt provided"}, 400

#         try:
#             outputs = generator(prompt)
#             return {
#                 "generated_text": outputs[0]["generated_text"]
#             }, 200
#         except Exception as e:
#             app.logger.error(f"Text generation failed: {e}")
#             return {"error": "Text generation failed"}, 500

# # === ADD ROUTES ===
# api.add_resource(Classify, "/api/classify")
# api.add_resource(Generate, "/api/generate")

# === RUN SERVER ===
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000, debug=True)


from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
from transformers import (
    pipeline,
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoTokenizer,
    AutoModelForCausalLM   # changed import
)
import os
import torch

app = Flask(__name__)
CORS(app)
api = Api(app)

# === MODEL CONFIGS ===
IMG_MODEL_ID  = "andupets/real-estate-image-classification"
TEXT_MODEL_ID = "openai-community/gpt2"
HF_TOKEN      = os.environ.get("HUGGINGFACE_TOKEN", None)
device        = 0 if torch.cuda.is_available() else -1

app.logger.info(f"Device set to use {'cuda' if device==0 else 'cpu'}")

# === IMAGE CLASSIFICATION SETUP ===
img_processor = AutoImageProcessor.from_pretrained(
    IMG_MODEL_ID,
    use_fast=True,
    use_auth_token=HF_TOKEN
)
img_model = AutoModelForImageClassification.from_pretrained(
    IMG_MODEL_ID,
    use_auth_token=HF_TOKEN
)
classifier = pipeline(
    task="image-classification",
    model=img_model,
    feature_extractor=img_processor,
    device=device,
    top_k=5
)

# === TEXT GENERATION SETUP (GPT-2) ===
text_tokenizer = AutoTokenizer.from_pretrained(
    TEXT_MODEL_ID, 
    use_auth_token=HF_TOKEN
)
text_model = AutoModelForCausalLM.from_pretrained(
    TEXT_MODEL_ID,
    use_auth_token=HF_TOKEN
)
generator = pipeline(
    task="text-generation",        # causal LM generation
    model=text_model,
    tokenizer=text_tokenizer,
    device=device,
    max_length=100,                # tune as needed
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# === HEALTH CHECK ===
@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok"), 200

# === IMAGE CLASSIFICATION ENDPOINT ===
class Classify(Resource):
    def post(self):
        data = request.get_json(force=True) or {}
        image_ref = data.get("image_url") or data.get("image_path")
        if not image_ref:
            return {"error": "No image URL or path provided"}, 400

        try:
            results = classifier(image_ref)
            return {"predictions": results}, 200
        except Exception as e:
            app.logger.error(f"Image classification failed: {e}")
            return {"error": "Image classification failed"}, 500

# === TEXT GENERATION ENDPOINT ===
class Generate(Resource):
    def post(self):
        data = request.get_json(force=True) or {}
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return {"error": "No prompt provided"}, 400

        try:
            outputs = generator(prompt)
            return {"generated_text": outputs[0]["generated_text"]}, 200
        except Exception as e:
            app.logger.error(f"Text generation failed: {e}")
            return {"error": "Text generation failed"}, 500

# === ADD ROUTES ===
api.add_resource(Classify, "/api/classify")
api.add_resource(Generate, "/api/generate")

# === RUN SERVER ===
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000, debug=True)
