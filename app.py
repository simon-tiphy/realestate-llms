# app.py

from flask import Flask, request
from flask_restful import Resource, Api
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import bitsandbytes as bnb

# -- 1. Initialize Flask + Flask‑RESTful --
app = Flask(__name__)
api = Api(app)

# -- 2. Load tokenizer & model at startup --
MODEL_NAME = "peft_realestate_llm"  # Path to your fine‑tuned model directory

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Model (4‑bit quantized with bitsandbytes)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    load_in_4bit=True,
    quantization_config=bnb.QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )
)

# Generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,           # Change to -1 for CPU, or use device_map="auto"
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# -- 3. Define the RESTful resource --
class Generate(Resource):
    def post(self):
        """
        POST /api/generate
        Body JSON: { "prompt": "Your question here" }
        Returns: { "generated_text": "...", "score": float }
        """
        payload = request.get_json(force=True)
        prompt = payload.get("prompt", "").strip()
        if not prompt:
            return {"error": "No prompt provided"}, 400

        # Run generation
        outputs = generator(prompt)
        first = outputs[0]

        return {
            "generated_text": first["generated_text"],
            "score": first.get("score")
        }, 200

# -- 4. Add resource to API --
api.add_resource(Generate, "/api/generate")

# -- 5. Start the server --
if __name__ == "__main__":
    # You can set debug=False in production
    app.run(host="0.0.0.0", port=8000, debug=True)
