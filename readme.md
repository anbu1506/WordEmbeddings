# 3D Word Embeddings

This project demonstrates how to create 3D word embeddings using PyTorch and visualize them using Plotly.

## Files

- `3d_word_embeddings.pth`: The saved model weights for the 3D word embeddings.
- `index.html`: An interactive 3D plot of the word embeddings.
- `readme.md`: This file.
- `sentences.txt`: A text file containing sentences used for training the word embeddings.
- `simple_3d_word_embedding.ipynb`: A Jupyter notebook that contains the code for training the word embeddings and generating the 3D plot.

## Requirements

- Python 3.10
- PyTorch
- Plotly

## Usage

1. **Install the required packages:**
    ```sh
    pip install torch plotly
    ```

2. **Run the Jupyter notebook:**
    Open [simple_3d_word_embedding.ipynb](https://anbu1506.github.io/WordEmbeddings/) in Jupyter Notebook or JupyterLab and run all cells to train the model and generate the 3D plot.

3. **View the 3D plot:**
    Open [index.html](https://anbu1506.github.io/WordEmbeddings/) in a web browser to view the interactive 3D plot of the word embeddings.

## Code Overview

- **Word Embedding Layer:**
    ```python
    class WordEmbeddingLayer(nn.Module):
        def __init__(self, vocab_size, embedding_dim):
            super(WordEmbeddingLayer, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)

        def forward(self, x):
            return self.linear(self.embedding(x).mean(dim=1))

        def get_embedding(self, idx):
            return self.embedding(torch.tensor([idx]))
    ```

- **Reading Data:**
    ```python
    def read_data(filename):
        with open(filename) as f:
            data = f.readlines()
            data = [line.replace('\n', '<eos>') for line in data]
        return data

    sentences = read_data("sentences.txt")
    ```

- **Tokenization:**
    ```python
    def tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())

    tokens = list(set(token for sentence in sentences for token in tokenize(sentence)))
    word2idx = {word: i for i, word in enumerate(tokens)}
    idx2word = {i: word for i, word in enumerate(tokens)}
    ```

- **Training:**
    ```python
    def train(epoch):
        for i in range(epoch):
            total_loss = 0
            for context, target in dataloader:
                output = wordEmbeddings(context)
                loss = criteria(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {i}, Loss: {total_loss / len(dataloader)}")

    train(3000)
    ```

- **Saving Model:**
    ```python
    torch.save(wordEmbeddings.state_dict(), "3d_word_embeddings.pth")
    ```

- **Visualization:**
    ```python
    fig = go.Figure()
    x, y, z, texts = [], [], [], []
    for i in range(len(tokens)):
        embedding = get_embedding_to_idx(i)
        num_emd = embedding.detach().numpy()
        x.append(num_emd[0][0])
        y.append(num_emd[0][1])
        z.append(num_emd[0][2])
        texts.append(idx2word[i])

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        text=texts,
        textposition='top center',
        marker=dict(size=5, color='red'),
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis'
        ),
        title="Interactive 3D Point Plot",
    )

    fig.show()
    fig.write_html("index.html")
    ```