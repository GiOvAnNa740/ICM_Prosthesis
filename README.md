# ICM_Prosthesis
Development of a prosthesis for the upper limb with tactile feedback connected to a brain-machine interface


DESENVOLVIMENTO DE INTERFACE CÉREBRO-MÁQUINA (ICM) PARA CONTROLE DE PRÓTESE TRANSRADIAL VIA AQUISIÇÃO SINAIS DE EEG

BRAIN-MACHINE INTERFACE (BMI) DEVELOPMENT TO CONTROL TRANSRADIAL PROSTHESIS VIA ACQUISITION OF EEG SIGNALS


Bruno Jesus dos Santos (Orientador)
Giovanna Lima Gomes Silva
Graduada em Ciência da Computação – USJT
Programa Ânima de Iniciação Científica – PROCIÊNCIA – 2022/2
(11) 94035-24567
giovanna1001.lima@gmail.com
 

#Resumo
A amputação de membro superior pode constituir impasses significativos na vida de uma pessoa tanto por questões estéticas quanto funcionais, e uma das opções disponíveis são as próteses mioelétricas, que utilizam aquisição e interpretação de padrões mioelétricos para o controle das funções motoras da prótese. O intuito é utilizar sinais originados de eletroencefalograma (EEG) com a utilização de técnicas de inteligência artificial (IA) para reconhecimento de padrões que se traduzam nas intenções motoras do usuário, de forma a controlar a prótese transradial. Existem diversos métodos de aprendizado de máquina que podem ser utilizados para tal fim, este artigo tem por objetivo demonstrar a tratativa e processamento de dados EEG originados de um dataset com técnicas de IA em Python para avaliar e comparar a precisão obtida com a finalidade de utilizá-los posteriormente para o controle de uma prótese trasradial.
Palavras-chave: Interface Cérebro-Máquina (ICM); Eletroencefalograma (EEG); Prótese trasradial

#Abstract
Upper limb amputation can constitute significant impasses in a person's life for both aesthetic and functional reasons, and one of the options available for people in such conditions are myoelectric prostheses, which use acquisition and interpretation of myoelectric patterns to control the prosthesis' motor functions. The intent is to use signals originating from electroencephalogram (EEG) with the use of artificial intelligence (AI) techniques to recognize patterns that translate into the user's motor intentions in order to control the transradial prosthesis. There are several machine learning methods that can be used for this purpose, this article aims to demonstrate the treatment and processing of EEG data originated from a dataset with AI techniques to evaluate and compare the accuracy obtained with the purpose of using them later for the control of a transradial prosthesis.
Keywords: Brain-Machine Interface (BMI); Electroencephalogram (EEG); Transradial prosthesis

 
#INTRODUÇÃO
A perda do membro superior pode impactar profundamente a vida de uma pessoa, sua autoestima, autonomia e capacidade de desenvolver tarefas do dia-a-dia. A amputação de membro superior é a remoção de qualquer extensão dos membros superiores (braço, antebraço, mãos, ou dedos) resultante de cirurgia, trauma, ou patologia.
A principal causa de amputações de membros superiores é o trauma/acidente, seguido de câncer, complicações associadas a doenças vasculares e anomalias congênitas (PEIXOTO, 2017). Desta última, constatou-se que as deficiências nos membros superiores eram significativamente mais comuns do que as deficiências nos membros inferiores, representados 73.3% dos casos analisados (RAND; VANODIA, 2013). 
Existem diferentes tipos de amputação e classificações possíveis relacionadas ao membro superior variando conforme o local da amputação, entretanto, este artigo visa o desenvolvimento de uma prótese para amputações do tipo transradial, que pode ocorrer em qualquer proporção entre o cotovelo e o pulso (GOLD; WESTGATE; HOLMES, 2011).

 
Figura 1 – Tipos de amputação do membro superior (BRASIL, 2014)

A utilização de próteses é uma das opções possíveis a pessoas em tais condições, e o sucesso do uso a longo prazo depende primariamente na percepção do usuário de conforto e serventia da prótese. Sendo adaptabilidade e personalização com bases nas necessidades de cada um alguns dos fatores chave para o sucesso e usabilidade continuada (JOSEPH, 2018).
O objetivo deste estudo é desenvolver uma prótese transradial com alguns movimentos pré-definidos controlados via aquisição e interpretação de sinais EEG com o uso de métodos de aprendizado de máquina em Python. Inicialmente os movimentos objetivados seriam apenas abrir e fechar a mão (não em punho, mas em “pinça” com pressão suficiente para ser capaz de segurar pequenos objetos).

#1.	OBTENÇÃO E PROCESSAMENTO DOS DADOS
Os dados para esta análise foram obtidos de uma base disponível na plataforma Kaggle, estes foram coletados utilizando o neuroheadset comercial “EMOTIV EPOC+ 14” da empresa EMOTIV para captura dos sinais EEG com uma frequência de amostragem de 128 Hz, com 14 canais de eletrodos: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8 e AF4. E Classificação de frequências em: theta, alpha, beta e delta (TORQUATO, 2019).
 
Figura 2 (Disponível em https://www.fieldtriptoolbox.org/faq/capmapping/) – Canais de eletrodos de sinais EEG

 
Figura 3 (Disponível em https://www.emotiv.com/epoc/) - EMOTIV EPOC+ 14

No experimento, foram coletados dados de 4 participantes diferentes em ciclos de 25 minutos, durante cada um dos ciclos, o participante foi exposto às imagens que representavam ações motoras voluntárias: uma seta para a direita que representaria a ação motora na direção direita, uma seta para a esquerda que representaria a ação motora na direção esquerda e um círculo que representaria nenhuma ação motora. Tais ações foram utilizadas para rotular os dados obtidos em “1.0”, “2.0” e “0.0”, respectivamente.
Os dados também passaram inicialmente por um pré-processamento no qual foi aplicado o algoritmo Transformada Rápida de Fourier (FFT), e depois foram calculadas as médias ponderada e aritmética de cada um dos 14 canais do dispositivo e para cada uma das 4 classificações de frequência de onda.
Iniciado o processo de tratativa e análise da base de dados em Python com a utilização da biblioteca Scikit-Learn, os dados obtidos de cada um dos 4 participantes foram consolidados em um único arquivo .csv. As colunas correspondentes às frequências e cada um dos canais, passaram por um processo de normalização com o uso da função “preprocessing.normalize”, enquanto que a coluna “Class”, equivalente aos rótulos das ações realizadas, foi mantida para categorização.
Inicialmente foi implementado o modelo de aprendizagem Support Vector Machine (SVM) (KOKATE; PANCHOLI; JOSHI, 2021). Foi então realizada a comparação de eficácia dos resultados obtidos com a utilização de quatro dataframes distintos criados a partir dos dados originais: Primeiro com dados de todos os transmissores, classificações de sinal e médias; segundo, com os dados apenas das médias aritméticas; terceiro com os dados apenas das médias ponderadas; e quarto, com os dados apenas das frequências beta, que são sinais diretamente associados a funções motoras (KUMAR; BHUVANESWARI, 2012). 
Comparando a acurácia de cada implementação, a variação de acerto foi de ± 2.04%, e a melhor performance atingiu apenas 50.04% de precisão, longe de um cenário ideal para utilização real, que seria uma porcentagem aproximada acima de 95% para uma utilização eficaz e confortável em pacientes.

 
Figura 4 - Matriz de Confusão SVM

#2.	CONSIDERAÇÕES FINAIS
Já está em progresso a aquisição direta de dados provenientes de voluntários utilizando o mesmo neuroheadset “EMOTIV EPOC”, para substituir a utilização de uma base de dados de terceiros, permitindo maior experimentação e liberdade para criar as condições de teste necessárias. Serão avaliados diversos cenários com variações de estímulos para melhor identificação dos sinais neurais.
Os próximos passos consistem na aplicação de métodos de classificação para reconhecer possíveis pares ou conjuntos de eletrodos e sinais EEG que demonstrem de maneira mais direta a intenção do usuário acerca do movimento motor para melhorar a precisão obtida com o aprendizado de máquina, além da avaliação e comparação de outros métodos como Redes Neurais Artificiais (ANN), Perceptron Multi-Camadas (MLP) e Análise Discriminante Linear (LDA).
Será também avaliada a necessidade de normalização dos dados obtidos em tempo real ou se tal pré-processamento pode comprometer o tempo de resposta da prótese.
Há de se considerar também que os dados obtidos são significativamente diferentes de voluntários amputados e não amputados, portanto os resultados obtidos de voluntários não amputados para a obtenção de frequências cerebrais podem apresentar uma menor eficácia quando colocado a teste com dados em tempo real de pessoas amputadas. 
Uma vez definidos os sinais e algoritmos a serem utilizados, a ICM será testada com dados em tempo real obtidos do neurotransmissor conectado a uma prótese transradial a ser desenvolvida. O objetivo é não apenas atingir uma acurácia acima de 95%, mas também garantir um tempo de resposta – desde a recepção do sinal, classificação e movimento da prótese – entre 200 e 400 milissegundos, geralmente aceito como o tempo máximo recomendado para utilização real (NAIDU et al., 2008). Portanto, faz-se necessário avaliar cada etapa do processo, individualmente e em conjunto, buscando otimizá-lo para atingir tais objetivos pretendidos


#REFERÊNCIAS
BRASIL. Técnico em órteses e próteses: livro-texto/ Ministério da Saúde. Secretaria de Gestão do Trabalho e da Educação na Saúde. Departamento de Gestão do Trabalho na Saúde – Brasília: Ministério da Saúde, 2014.
CORDELLA, Francesca et al. Literature review on needs of upper limb prosthesis users. Frontiers in neuroscience, v. 10, p. 209, 2016. https://doi.org/10.3389/fnins.2016.00209.
CUNHA, Fransérgio L. et al. O uso de redes neurais artificiais para o reconhecimento de padrões em uma prótese mioelétrica de mão. In: VIII Congresso Brasileiro de Redes Neurais. 2007.
DRUMOND, Marina Silva Bueno. Brain machine interface (BMI) para próteses artificiais: sub-sistema de aquisição de sinais cerebrais. 2021. 
GOLD, Nina B.; WESTGATE, Marie‐Noel; HOLMES, Lewis B. Anatomic and etiological classification of congenital limb deficiencies. American journal of medical genetics Part A, v. 155, n. 6, p. 1225-1235, 2011. https://doi.org/10.1002/ajmg.a.33999
GULER, Inan; UBEYLI, Elif Derya. Multiclass support vector machines for EEG-signals classification. IEEE transactions on information technology in biomedicine, v. 11, n. 2, p. 117-126, 2007. https://doi.org/10.1109/TITB.2006.879600
JOSEPH, Burris. Braddom's Rehabilitation Care: A Clinical Handbook. Elsevier Health Sciences, 2018. https://doi.org/10.1016/B978-0-323-47904-2.00009-X
KOKATE, Pranali; PANCHOLI, Sidharth; JOSHI, Amit M. Classification of upper arm movements from eeg signals using machine learning with ica analysis. arXiv preprint arXiv:2107.08514, 2021. https://doi.org/10.48550/arXiv.2107.08514
KUMAR, J. Satheesh; BHUVANESWARI, P. Analysis of Electroencephalography (EEG) signals and its categorization–a study. Procedia engineering, v. 38, p. 2525-2536, 2012. https://doi.org/10.1016/j.proeng.2012.06.298
NAIDU, D. Subbaram et al. Control strategies for smart prosthetic hand technology: An overview. In: 2008 30th Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE, 2008. p. 4314-4317. https://doi.org/10.1109/IEMBS.2008.4650164
PARAJULI, Nawadita et al. Real-time EMG based pattern recognition control for hand prostheses: a review on existing methods, challenges and future implementation. Sensors, v. 19, n. 20, p. 4596, 2019. https://doi.org/10.3390/s19204596
PEIXOTO, Alberto Monteiro et al. Prevalência de amputações de membros superiores e inferiores no estado de Alagoas atendidos pelo SUS entre 2008 e 2015. Fisioterapia e Pesquisa, v. 24, p. 378-384, 2017. https://doi.org/10.1590/1809-2950/17029524042017
RAND, Stephanie; VANODIA, Vinay. Upper Limb Amputations. 2013. https://now.aapmr.org/upper-limb-amputations
SIMON, Ann M. et al. The target achievement control test: Evaluating real-time myoelectric pattern recognition control of a multifunctional upper-limb prosthesis. Journal of rehabilitation research and development, v. 48, n. 6, p. 619, 2011. https://doi.org/10.1682/jrrd.2010.08.0149
SUBASI, Abdulhamit; ERCELEBI, Ergun. Classification of EEG signals using neural network and logistic regression. Computer methods and programs in biomedicine, v. 78, n. 2, p. 87-99, 2005. https://doi.org/10.1016/j.cmpb.2004.10.009
TORQUATO, Fabricio.“EEG data from hands movement” [Data set] Kaggle, 2019 https://doi.org/10.34740/KAGGLE/DS/391999
ZHANG, Xiao Lei et al. Event related potentials during object recognition tasks. Brain research bulletin, v. 38, n. 6, p. 531-538, 1995. https://doi.org/10.1016/0361-9230(95)02023-5
