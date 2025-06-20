import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import transformers
    import torch
    return (torch,)


@app.cell
def _(mo):
    mo.md(r"""## Sentiment classifier""")
    return


@app.cell
def _():
    from transformers import pipeline


    return (pipeline,)


@app.cell
def sentimentanalysermodel(pipeline):
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_classifier = pipeline(model=model_name)

    text_input = "I'm really excited about using Hugging Face to run AI models!"
    print(sentiment_classifier(text_input))

    text_input = "I'm having a horrible day today."
    print(sentiment_classifier(text_input))

    text_input = "Most of the Earth is covered in water."
    print(sentiment_classifier(text_input))

    text_input_list = [
        "I'm really excited about using Hugging Face to run AI models!",
        "I'm having a horrible day today.",
        "Most of the Earth is covered in water."
    ]
    print(*sentiment_classifier(text_input_list), sep='\n')

    return (model_name,)


@app.cell
def _(mo):
    mo.md(r"""## Zero shot classifier""")
    return


@app.cell
def _(pipeline):
    model_name_2 = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    zs_text_classifier = pipeline("zero-shot-classification",model=model_name_2)

    candidate_labels = [
         "Billing Issues",
         "Technical Support",
         "Account Information",
         "General Inquiry",
    ]
    hypothesis_template = "This text is about {}"


    customer_text = "My account was charged twice for a single order."
    print(zs_text_classifier(
        customer_text,
        candidate_labels,
        hypothesis_template=hypothesis_template,
        multi_label=True
    ))

    customer_text = "I need help with configuring my new wireless router please "
    print(zs_text_classifier(
        customer_text,
        candidate_labels,
        hypothesis_template=hypothesis_template,
        multi_label=True
    ))


    return


@app.cell
def _(mo):
    mo.md(r"""## Zero shot image classifier""")
    return


@app.cell
def _(pipeline):
    """
    No model specified,so uses default model
    """
    image_classifier = pipeline(task="image-classification")
    print(f"The default model picked by pipeline is {image_classifier.model.name_or_path}")

    predictions = image_classifier(["farm-animal-portraits-rob-macinnis-fb.png"])

    print(f"Predictions : \n {predictions}",end="\n\n")

    print(f"Total Predictions : {len(predictions[0])}")
    for prediction in predictions[0] :
        print(f"{prediction}")

    print("\n")

    predictions = image_classifier(["kuvasz-dog-breed-full-body.webp"])
    for prediction in predictions[0] :
        print(f"{prediction}")

    print("\n")

    predictions = image_classifier(["Musk-ox.jpg.webp"])
    for prediction in predictions[0] :
        print(f"{prediction}")

    print("\n")

    predictions = image_classifier(["images.jpeg"])
    for prediction in predictions[0] :
        print(f"{prediction}")
    return


@app.cell
def _(mo):
    mo.md(r"""## Fine tuning""")
    return


@app.cell
def _(model_name):
    # Explore current 

    from transformers import AutoTokenizer

    # same model_name from cell http://localhost:2718/#scrollTo=sentimentanalysermodel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_text = "I really want to go to an island. Do you want to go?"

    # Tokenize
    encoded_input = tokenizer(input_text)
    print(encoded_input, end="\n\n")
    print(encoded_input["input_ids"], end="\n\n")

    # Reconvert token to text
    print(tokenizer.convert_ids_to_tokens(encoded_input["input_ids"]), end="\n\n")

    print(f"Current vocabulary size for {model_name} : {tokenizer.vocab_size}")

    return (tokenizer,)


@app.cell
def _(tokenizer):
    # Add new tokens

    new_tokens = [
        "Garuda",
        "whaleshark"
    ]

    # Token ID 3 corresponds to <unk>, which is the default token for input tokens that aren’t in the vocabulary.
    print(tokenizer.convert_tokens_to_ids(new_tokens), end="\n\n")
    print(f"{tokenizer.convert_ids_to_tokens(3)}, {tokenizer.convert_ids_to_tokens(50265)}")

    print(tokenizer.add_tokens(new_tokens))

    print(f"{tokenizer.convert_tokens_to_ids(new_tokens)}")
    return


@app.cell
def _(model_name):
    # load the model object directly using AutoModelForSequenceClassification

    from transformers import AutoModelForSequenceClassification

    model_details = AutoModelForSequenceClassification.from_pretrained(model_name)
    print(model_details,end="\n\n")
    print(model_details.roberta.embeddings)
    return (model_details,)


@app.cell
def _(model_details, model_name, tokenizer, torch):
    # Full pipeline


    # import torch
    # from transformers import (
    #     AutoTokenizer,
    #     AutoModelForSequenceClassification,
    #     AutoConfig
    # )

    # model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # config = AutoConfig.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # text = "I love using the Transformers library!"
    # encoded_input = tokenizer(text, return_tensors="pt")

    # with torch.no_grad():
    #     output = model(**encoded_input)

    # scores = output.logits[0]
    # probabilities = torch.softmax(scores, dim=0)

    # for i, probability in enumerate(probabilities):
    #     label = config.id2label[i]
    #     print(f"{i+1}) {label}: {probability}")

    """
    Above same as this code : 
        from transformers import pipeline
    
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        text = "I love using the Transformers library!"
    
        full_pipeline = pipeline(model=model_name)
        full_pipeline(text)
    """

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name)

    # Define text and tokenize it with tokenizer(). 
    text = "I love using the Transformers library!"
    encoded_input_for_pipeline = tokenizer(text, return_tensors="pt")

    # Then pass the tokenized input, encoded_input, to the model object and store the results as output
    # Speed up model inference by disabling gradient calculations.
    with torch.no_grad():
        output = model_details(**encoded_input_for_pipeline)

    scores = output.logits[0]
    probabilities = torch.softmax(scores, dim=0)

    for i, probability in enumerate(probabilities):
        label = config.id2label[i]
        print(f"{i+1}) {label}: {probability}")
    return


if __name__ == "__main__":
    app.run()
