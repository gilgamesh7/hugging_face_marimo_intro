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
    return


@app.cell
def _(mo):
    mo.md(r"""## Sentiment classifier""")
    return


@app.cell
def _():
    from transformers import pipeline


    return (pipeline,)


@app.cell
def _(pipeline):
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

    return


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


if __name__ == "__main__":
    app.run()
