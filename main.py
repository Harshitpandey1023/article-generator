from flask import Flask, render_template, request
import gpt_2_simple as gpt2
import tensorflow as tf

app = Flask(__name__)
sess = None

def download_gpt2_model(model_name="124M"):
    gpt2.download_gpt2(model_name=model_name)

def load_gpt2_model(model_name="124M"):
    global sess
    if sess is None:
        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(sess, model_name=model_name)

def generate_article(topic, length=300, temperature=0.7):
    global sess
    article = gpt2.generate(
        sess,
        prefix=f"Topic: {topic}\n\nIn this article, we will discuss",
        length=length,
        temperature=temperature,
        return_as_list=True
    )[0]
    return article

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        topic = request.form['topic']
        generated_article = generate_article(topic)
        return render_template('index.html', topic=topic, generated_article=generated_article)
    return render_template('index.html', topic=None, generated_article=None)

def main():
    # Download and load the GPT-2 model
    download_gpt2_model()
    load_gpt2_model()

    # Run the Flask app in debug mode
    app.run(debug=True)

if __name__ == "__main__":
    main()
    