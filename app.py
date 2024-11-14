from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import set_seed
from utils import gen
import pickle

app = Flask(__name__)
CORS(app)

seed = 42
set_seed(seed)

# Load the model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/generate", methods=["POST"])
def generate_text():
    # Parse input JSON
    data = request.get_json()
    query = data.get("query", "")

    # Build prompt and generate response
    prompt = f"Instruct: Answer the following question in the context of DevOps model practices.\n{query}\nOutput:\n"
    generated_res = gen(model, prompt, maxlen=200)
    peft_model_output = generated_res[0].split("Output:\n")[1]
    prefix, success, result = peft_model_output.partition("###")

    # Format the response
    response = {"input_prompt": prompt, "generated_summary": prefix.strip()}
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
