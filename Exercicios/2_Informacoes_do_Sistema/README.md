# Exercício sobre informações de sistema de computador e Google Colab

**Entrega até 23h59 de 09/03/2021**.

**Exercício individual para ser feito em Jupyter Notebook local e rodando em Google Colab.**

Vide o [diretório "Informações sobre Sistema"](https://github.com/rcolistete/Ferramentas_Ensino_Pesquisa/tree/main/Informacoes_Sistema) do repositório público ["Ferramentas de Ensino e Pesquisa"](https://github.com/rcolistete/Ferramentas_Ensino_Pesquisa/) , e use :

**a) um dos arquivos, "Informacoes_Sistema.ipynb" (se usar Windows) ou "Informacoes_Sistema_LinuxMacOS.ipynb" (se usar Linux/Mac OS) para rodar localmente no seu computador**, em instalação local do Anaconda (citado desde a 1a aula de CEF-RC). Adicione após o título/subtítulos um parágrafo resumindo a configuração do seu computador. Por exemplo :

    Computador de FULANO :
    Notebook Dell XPS 15 com CPU Intel Core i7 2670QM, 2,2-3,1 GHz 4 cores/8 threads 6MB cache 
    com GPU integrada Intel HD Graphics 3000, 
    GPU dedicada NVidia GeForce GPU GT 540M, 2GB, 96 cores CC2.1, 8GB de RAM, HD de 1 TB.
    Linux Manjaro KDE 18.0 64 bits.

**Renomeie tal arquivo para formato "Informacoes_Sistema_NOMECOMPUTADOR.ipynb"**, onde é NOMECOMPUTADOR o nome abreviado do seu computador (p. e., DellXPS15, se fosse o notebook acima), sem espaços, mas pode usar \'_'. **Crie também versão .html via exportação**;

**b) "Informacoes_Sistema_LinuxMacOS.ipynb" para rodar no Google Colab. Renomeie tal arquivo para "Informacoes_Sistema_GoogleColab.ipynb". Crie também versão .html via exportação.**

**Crie então um documento Jupyter Notebook com nome "Exercicio_CEF-RC_InformacoesSistema_NOME_DATA.ipynb"** (onde NOME é o nome do aluno, sem espaços, mas pode usar '\_', e DATA é data, sem espaços, p. e., 2021-03-09) com :

- nome do exercício, da disciplina, seu nome, data de entrega, etc, no título, subtítulo, etc;
- uma seção com título "Versões de Python e módulos";
- nessa seção crie uma tabela (em Markdown) com os seguintes títulos de colunas :
	* Python/Módulo;
	* Anaconda;
	* Computador;
	* Google Colab;

- onde as colunas :
	* Python/Módulo tem como linhas Python, NumPy, MatPlotLib, SymPy, Pandas, Bokeh, Holoviews, Seaborn, Numba e CuPy;
	* Anaconda tem as versões de Python e de cada módulo da 1a coluna, vide página ["Anaconda package lists"](https://docs.anaconda.com/anaconda/packages/pkg-docs/) (escolha seu sistema operacional e Python mais recente) :
	* Computador tem as versões de Python e cada módulo obtidos após rodar (a). Se algum módulo não estiver instalado, coloque '\-';
	* Google Colab tem as versões de Python e cada módulo obtidos após rodar (b);

- ao final da seção escreva comparando :
	* quais das 3 opções da tabela têm mais módulos instalados e versões mais recentes.
- crie outra seção comparando hardware do seu computador e do Google Colab.


**Enviar ao professor** o arquivo "Exercicio_CEF-RC_InformacoesSistema_NOME_DATA.ipynb" e o arquivo da versão exportada .html respectiva, juntamente com os arquivos criados em (a) e (b) :
- dentro de arquivo compactado .zip de um diretório "Exercicio_CEF-RC_InformacoesSistema_NOME_DATA" (contendo os arquivos), **via essa atividade**;
- **via repositório privado no GitHub do aluno**, criado com nome "CEF_Atividades", adicionando o professor ("rcolistete") como colaborador, dentro do repositório crie diretório "Exercicio_InformacoesSistema" e coloque os arquivos .ipynb e .html.