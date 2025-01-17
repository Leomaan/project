import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score

# Definir o classificador
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

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Preparar Dataset
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
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove a dimensão extra
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx])
        }

# Função de treinamento
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

# Função de avaliação
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

# Função para treinar e testar
def train_and_predict():
    sentences = []  # Frases fornecidas
    labels = []  # Rótulos para as frases (0 = não irônica, 1 = irônica)
    
    # Definir o modelo
    model = IronyClassifier()

    # Verificar se há um modelo salvo
    model_path = "irony_classifier_model.pth"
    try:
        model.load_state_dict(torch.load(model_path))
        print("Modelo carregado com sucesso!")
    except FileNotFoundError:
        print("Nenhum modelo salvo encontrado. Treinando um novo modelo...")

    # Configurar otimizador e função de perda
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    while True:
        # Entrada da frase
        new_sentence = input("Digite uma frase (ou 'sair' para encerrar): ").strip()
        if new_sentence.lower() == 'sair':
            break

        # Entrada do rótulo (0 para não irônica, 1 para irônica) com validação
        while True:
            try:
                label = int(input("Digite o rótulo (0 para não irônica, 1 para irônica): ").strip())
                if label not in [0, 1]:
                    print("O rótulo deve ser 0 ou 1. Tente novamente.")
                    continue
                break
            except ValueError:
                print("Entrada inválida! Digite 0 ou 1.")
        
        # Adicionar a nova frase e rótulo
        sentences.append(new_sentence)
        labels.append(label)

        # Criar Dataset e DataLoader
        dataset = IronyDataset(sentences, labels, tokenizer)
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

        # Treinar o modelo com as novas frases
        train_model(model, train_loader, optimizer, criterion)
        
        # Avaliar o modelo
        accuracy = evaluate_model(model, train_loader)
        
        # Previsão de uma frase para mostrar o desempenho do modelo
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

        # Salvar o modelo após cada iteração
        torch.save(model.state_dict(), model_path)
        print(f"Modelo salvo em {model_path}")

# Executar a função
train_and_predict()
