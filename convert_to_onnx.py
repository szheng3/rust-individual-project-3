import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if __name__ == '__main__':

    # Define model and tokenizer names
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer_name = model_name

    # Download the pre-trained model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Set model to evaluation mode and convert it to ONNX
    model.eval()
    dummy_input = torch.tensor(tokenizer.encode("Hello, world!", add_special_tokens=True)).unsqueeze(0)
    torch.onnx.export(model, dummy_input, "summary_model.onnx", opset_version=12, input_names=["input"], output_names=["output"])

    print("Model converted to ONNX and saved as summary_model.onnx")
