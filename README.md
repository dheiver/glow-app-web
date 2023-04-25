Classificador de Tumores de Mama
Este é um aplicativo desenvolvido com o Streamlit que permite aos usuários fazer o upload de uma imagem de um tumor de mama e classificá-lo como benigno, normal ou maligno usando um modelo pré-treinado.

Como usar o aplicativo
Clone este repositório em seu ambiente local usando o comando git clone https://github.com/seu-usuario/classificador-tumores.git.
Instale as dependências necessárias listadas no arquivo requirements.txt usando o comando pip install -r requirements.txt.

Execute o aplicativo usando o comando streamlit run app.py.
No navegador que será aberto, faça o upload de uma imagem de um tumor de mama no formato .jpg, .jpeg ou .png.
O aplicativo irá classificar a imagem e exibir a classe predita e a probabilidade.

O aplicativo também exibirá três imagens lado a lado em três colunas. A primeira imagem mostra a imagem de entrada original, a segunda imagem mostra uma imagem binária onde os valores de pixel maiores ou iguais a 0,5 são definidos como 1 e os valores de pixel menores que 0,5 são definidos como 0, e a terceira imagem mostra a imagem binária após uma operação morfológica de abertura ter sido aplicada para remover pixels pequenos e preencher lacunas internas.
Arquivos do projeto

app.py: arquivo principal do aplicativo desenvolvido com o Streamlit.
BreastCancerSegmentor.h5: modelo pré-treinado usado pelo aplicativo.
requirements.txt: arquivo de dependências do Python necessárias para executar o aplicativo.

Requisitos
Python 3.6 ou superior. 
Streamlit
TensorFlow
NumPy
Pillow
OpenCV

Como contribuir
Faça um fork deste repositório.

Crie uma nova branch com a sua funcionalidade usando o comando git checkout -b minha-funcionalidade.
Faça as alterações necessárias e adicione os arquivos modificados usando o comando git add nome-do-arquivo.
Faça o commit das suas alterações usando o comando git commit -m "Minha mensagem de commit".
Faça o push da branch usando o comando git push origin minha-funcionalidade.
Envie um pull request com as suas alterações.
