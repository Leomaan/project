
Claro! Aqui está o conteúdo do README.md convertido em um único texto contínuo para você colocar no arquivo:

Classificador de Ironia com BERT
Este projeto utiliza o modelo BERT (Bidirectional Encoder Representations from Transformers) para classificar frases como "irônicas" ou "não irônicas". O modelo é treinado interativamente, permitindo que o usuário insira novas frases e rótulos durante o treinamento. Após cada iteração, o modelo é salvo, permitindo que o treinamento continue a partir de onde parou.

Funcionalidades
Classificação de Ironia: O modelo classifica as frases como "irônicas" ou "não irônicas".
Treinamento interativo: O modelo permite ao usuário adicionar novas frases e rótulos durante o processo de treinamento.
Treinamento com BERT: O modelo usa o BERT para representação de texto e uma camada totalmente conectada para a classificação final.
Modelo salvo: O modelo treinado é salvo em um arquivo irony_classifier_model.pth, podendo ser carregado para continuar o treinamento ou fazer previsões.
Requisitos
Para rodar o código, você precisará de:

Python 3.7+
PyTorch 1.8+
Transformers (biblioteca Hugging Face)
scikit-learn
Instale as dependências necessárias com:

bash
Copiar
pip install torch transformers scikit-learn
Estrutura do Código
1. Classificador de Ironia (IronyClassifier)
A classe IronyClassifier define o modelo, que é composto por:

BERT: Utiliza o modelo pré-treinado bert-base-uncased da Hugging Face para extrair representações de frases.
Camada totalmente conectada: A camada fc mapeia a saída do BERT para a classificação binária (irônica ou não irônica).
python
Copiar
class IronyClassifier(nn.Module):
    def __init__(self):
        super(IronyClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 2)  # 768 é o tamanho da saída do BERT, 2 classes: irônica ou não

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Representação da frase
        logits = self.fc(pooled_output)  # Classificação
        return logits
2. Tokenizer
Utiliza o BertTokenizer para tokenizar as frases antes de passá-las ao modelo. O tokenizador prepara as frases para que o modelo possa entender, realizando padding e truncamento conforme necessário.

python
Copiar
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
3. Preparação do Dataset (IronyDataset)
A classe IronyDataset é usada para estruturar o dataset de treinamento. Cada frase é tokenizada, e as sequências tokenizadas, junto com as máscaras de atenção e os rótulos, são preparadas para o treinamento.

python
Copiar
class IronyDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.sentences[idx], truncation=True, padding='max_length', max_length=64, return_tensors="pt")
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx])
        }
4. Funções de Treinamento e Avaliação
A função train_model realiza o treinamento do modelo, enquanto evaluate_model avalia a acurácia do modelo em um conjunto de dados.

python
Copiar
def train_model(model, train_loader, optimizer, criterion):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate_model(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, dim=1)
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy
5. Treinamento Interativo (train_and_predict)
A função principal train_and_predict permite que o usuário insira frases e rótulos interativamente. O modelo é treinado a cada iteração e salva automaticamente após cada ciclo.

python
Copiar
def train_and_predict():
    sentences = []  # Frases fornecidas
    labels = []  # Rótulos para as frases (0 = não irônica, 1 = irônica)
    
    model = IronyClassifier()

    # Verificar se há um modelo salvo
    model_path = "irony_classifier_model.pth"
    try:
        model.load_state_dict(torch.load(model_path))
        print("Modelo carregado com sucesso!")
    except FileNotFoundError:
        print("Nenhum modelo salvo encontrado. Treinando um novo modelo...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    while True:
        new_sentence = input("Digite uma frase (ou 'sair' para encerrar): ").strip()
        if new_sentence.lower() == 'sair':
            break

        while True:
            try:
                label = int(input("Digite o rótulo (0 para não irônica, 1 para irônica): ").strip())
                if label not in [0, 1]:
                    print("O rótulo deve ser 0 ou 1. Tente novamente.")
                    continue
                break
            except ValueError:
                print("Entrada inválida! Digite 0 ou 1.")
        
        sentences.append(new_sentence)
        labels.append(label)

        dataset = IronyDataset(sentences, labels, tokenizer)
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

        train_model(model, train_loader, optimizer, criterion)
        
        accuracy = evaluate_model(model, train_loader)
        
        test_sentence = input("Digite uma frase para testar a previsão: ").strip()
        encoding = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            output = model(encoding['input_ids'], encoding['attention_mask'])
            _, predicted = torch.max(output, dim=1)
            prediction = "irônica" if predicted.item() == 1 else "não irônica"
        
        print(f"Frase: '{test_sentence}'")
        print(f"Predição: {prediction}")
        print(f"Acurácia no treinamento: {accuracy:.2f}")
        print("-" * 50)

        torch.save(model.state_dict(), model_path)
        print(f"Modelo salvo em {model_path}")
Como Usar
Clone o repositório:

bash
Copiar
git clone https://github.com/Leomaan/project.git
cd project
Instale as dependências:

bash
Copiar
pip install torch transformers scikit-learn
Execute o código:

bash
Copiar
python train_and_predict.py
Insira frases e rótulos para treinar o modelo interativamente. O modelo será salvo a cada iteração, e você poderá continuar o treinamento mais tarde.

Contribuindo
Sinta-se à vontade para abrir issues e pull requests. Para contribuir, basta fazer um fork deste repositório e enviar suas alterações.

Licença
Este projeto está licenciado sob a MIT License - consulte o arquivo LICENSE para mais detalhes.