# Classificador de Ironia com BERT

Este projeto utiliza o modelo BERT (Bidirectional Encoder Representations from Transformers) para classificar frases como "irônicas" ou "não irônicas". O modelo é treinado interativamente, permitindo que o usuário insira novas frases e rótulos durante o treinamento. Após cada iteração, o modelo é salvo, permitindo que o treinamento continue a partir de onde parou.

## Funcionalidades

- **Classificação de Ironia:** O modelo classifica as frases como "irônicas" ou "não irônicas".
- **Treinamento interativo:** O modelo permite ao usuário adicionar novas frases e rótulos durante o processo de treinamento.
- **Treinamento com BERT:** O modelo usa o BERT para representação de texto e uma camada totalmente conectada para a classificação final.
- **Modelo salvo:** O modelo treinado é salvo em um arquivo `irony_classifier_model.pth`, podendo ser carregado para continuar o treinamento ou fazer previsões.

## Requisitos

Para rodar o código, você precisará de:

- Python 3.7+
- PyTorch 1.8+
- Transformers (biblioteca Hugging Face)
- scikit-learn

Instale as dependências necessárias com:

```bash
pip install torch transformers scikit-learn


