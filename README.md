# Eye Cat v3.2

> Detecção e segmentação em tempo real de gatos filhotes em uma câmera RTSP, com alerta de "fuga de gato".

## Sobre o projeto

Esse é um trabalho da matéria de Laboratório de Programação Orientada a Objetos, ministrada pelo professor Derzu Omaia, do curso de Engenharia da Computação da UFPB.

O trabalho consistia em criar qualquer programa com visão computacional em C++. Eu queria fazer algo mais complexo, então me dediquei mais para esse trabalho específico, já que eu já tinha experiência com outro trabalho que envolvia OpenCV.

## Motivação

No momento que recebi a tarefa, minha gata Marie tinha acabado de dar cria, e não era a primeira vez que ela tinha tido filhotes. Na última vez, um deles acabou fugindo da caixa e acabou se acidentando e, infelizmente, falecido.

Para ajudar a prevenir isso, tive a ideia de fazer um programa com uma câmera wifi que detectasse fugas de gatos da caixa em que eles estavam.

## Hardware

Comprei uma câmera wifi "Orbtronic ORB7160" na Amazon, uma câmera simples de R$ 60 que tinha suporte a wifi e alimentação via USB-C.

Quando chegou, notei que ela já tinha um software próprio, o "YooSee", que eu achei bem ruim e não me dava muita liberdade de personalização. Mas, por se tratar de uma câmera wifi, eu conseguia configurá-la para acessá-la apenas com o seu IP local e uma senha definida por mim mesmo.

Posicionei ela em uma grade que ficava em cima da caixa dos gatos filhotes, dessa forma tinha uma visão completa de todos os gatos e da região exterior da caixa. Conectei um powerbank de 10.000mAh que conseguia alimentá-la por um período de tempo próximo de 24 horas.

Com isso, tinha todo o hardware necessário para o programa funcionar e podia começar a trabalhar no software — aí que veio o primeiro obstáculo.

## Desafios e escolha da abordagem

Como você pode ver em algumas imagens do dataset, os gatos ficavam muito em cima uns dos outros, eram 6 gatos no total. Minha ideia inicial era tocar um alarme toda vez que o programa detectasse uma média de menos de 6 gatos na caixa por um período X, mas se um gato estivesse dormindo embaixo de outros gatos, ficando completamente invisível pela câmera, o programa poderia classificar isso como uma falsa fuga.

Outro problema que eu previ que aconteceria seria que, quando os gatos ficassem dormindo emaranhados, um por cima dos outros, ficaria muito difícil para os modelos convencionais de OpenCV identificar os gatos. Os modelos de Bounding Boxes, que são extremamente comuns, não servem bem nesse caso, uma vez que não dá para selecionar uma caixa que contenha apenas um gato, e muitas vezes o gato não aparecia por completo na câmera — aparecia só o rabo, ou só a cabeça — e eu sabia que um modelo de Bounding Boxes não seria o suficiente.

Eu testei modelos já treinados de Bounding Boxes e não consegui fazer nenhum funcionar para a minha situação: muitos deles detectavam mais de 10 gatos ou nenhum gato, e mesmo os modelos de segmentação não estavam funcionando. Passei muito tempo tentando usar um modelo já treinado, até que percebi que seria melhor eu mesmo treinar um modelo apenas para a minha situação.

Tive alguns problemas na hora de fazer e configurar o dataset, mas no final, utilizando boas práticas de criação e treinamento na plataforma da Ultralytics, consegui chegar em um resultado satisfatório.

## Construção do dataset

Para treinar um modelo em cima do dataset, tive que adotar algumas estratégias. No início, estava cometendo erros que enviesavam o modelo: as fotos que eu utilizei no início foram de apenas um dia, com o mesmo pano que estava sendo usado na caixa, os mesmos objetos, a mesma iluminação, e com "deformações/dobras" no pano parecidas. Isso fazia com que, se o modelo trabalhasse em cima de um padrão diferente de dobra de pano, acabasse detectando um gato em uma dobra de pano que se parecesse com um rabo ou uma pata.

Outro erro que cometi foi colocar apenas imagens que tinham gatos filhotes e imagens que sempre tinham a mesma quantidade de gatos filhotes, então o modelo poderia alucinar e pensar que sempre teria gatos filhotes na imagem, ou que sempre deveria haver 6 gatos na imagem.

No final, fiz um dataset sólido com mais de 450 imagens, corrigindo todos os erros que cometi anteriormente.

## Treinamento do modelo

O Ultralytics me devolveu um modelo treinado com as seguintes métricas:

| Métrica | Valor |
|---|---|
| mAP50(M) | 99.3% |
| mAP50-95(M) | 81.8% |
| Precision(M) | 98.1% |
| Recall(M) | 97.9% |

De forma breve, o que cada uma dessas métricas significa (o "(M)" indica que são métricas calculadas em cima da máscara de segmentação, não só da bounding box):

- **Precision (Precisão)**: de tudo que o modelo marcou como "gato filhote", qual porcentagem realmente era um gato filhote. Precisão alta significa poucos falsos positivos (o modelo não fica "vendo gato" onde não tem).
- **Recall (Revocação)**: de todos os gatos filhotes que realmente estavam na imagem, qual porcentagem o modelo conseguiu encontrar. Recall alto significa poucos falsos negativos (o modelo não deixa gato passar batido).
- **mAP50**: a precisão média do modelo considerando "acerto" quando a máscara prevista tem pelo menos 50% de sobreposição (IoU) com a máscara real. É a métrica padrão para avaliar o quão bem o modelo localiza os objetos.
- **mAP50-95**: a mesma ideia do mAP50, mas calculada em vários níveis de exigência de sobreposição (de 50% até 95%, em passos) e tirando a média. É uma métrica bem mais rigorosa, porque exige que a máscara prevista seja mais precisa geometricamente, não só "mais ou menos em cima" do gato.

## Conversão do modelo para NCNN

O modelo treinado saiu da Ultralytics no formato `exp.pt`, que é um arquivo de pesos do PyTorch. O problema é que o PyTorch não é feito para isso: ele é pesado, depende de bibliotecas Python, de CUDA (ou ROCm) para acelerar por GPU, e não roda bem em C++ puro num loop de vídeo em tempo real.

Como o programa final precisava rodar em C++, com baixa latência, e acelerado por GPU numa AMD RX6650XT (que não tem suporte oficial a CUDA e tem suporte limitado via ROCm), a solução foi usar o **NCNN**, um motor de inferência leve, em C++, feito pela Tencent, que roda via **Vulkan** — funcionando em qualquer GPU com driver Vulkan, incluindo placas AMD.

O problema é que o NCNN não lê arquivos `.pt` diretamente. Foi necessário converter o modelo em duas etapas:

1. Exportar o `exp.pt` para um formato intermediário (TorchScript/ONNX), usando a própria ferramenta de exportação do Ultralytics;
2. Converter esse formato intermediário para o formato nativo do NCNN usando o **PNNX** (PyTorch Neural Network eXchange), uma ferramenta feita pelo mesmo autor do NCNN, que faz essa ponte.

O resultado dessa conversão é a pasta `exp_ncnn_model/`, com os arquivos:

- `model.ncnn.param` — a arquitetura da rede (as camadas, como elas se conectam)
- `model.ncnn.bin` — os pesos treinados em si
- `metadata.yaml` e `model_ncnn.py` — gerados automaticamente pelo processo de exportação, com metadados do modelo

São esses dois primeiros arquivos (`.param` e `.bin`) que o `mainT.cpp` carrega diretamente para rodar a inferência.

## Funcionalidades

### Detecção e segmentação em tempo real

O núcleo do programa: a cada frame recebido da câmera via RTSP, o modelo treinado detecta cada gato filhote e desenha o contorno exato dele (segmentação), não apenas uma caixa. Um rastreador temporal confirma a detecção ao longo de alguns frames seguidos antes de contar/desenhar o gato na tela, evitando "alucinações" de um frame isolado.

### Alerta de fuga de gato

A funcionalidade principal. Toda vez que o programa detecta que há um gato fora da caixa monitorada e a meta de gatos "dentro" ainda não foi atingida, ele:

- Mostra um aviso piscando na tela ("FUGA DE GATO!");
- Toca um bipe repetido enquanto o alerta estiver ativo;
- Aciona o `toque.sh`, que utiliza uma ferramenta do KDE Connect (GSConnect) para fazer meu celular tocar.

Um dos problemas que eu tive foi que essa notificação só era enviada caso meu celular estivesse na mesma rede que o computador que estava rodando o programa. Resolvi isso de forma simples, fazendo uma conexão do meu celular ao computador via **ZeroTier**, então a notificação pode ser enviada mesmo se eu estiver fora de casa.

### Tirar foto

Pressionando a tecla **'f'** com a janela do programa em foco, é salva uma foto "pura" da câmera — ou seja, o frame original, sem nenhuma marcação de detecção desenhada em cima. O objetivo é poder guardar registros limpos dos gatos (para o dataset, para acompanhar o crescimento deles, etc.) sem precisar segurar o celular, direto do próprio programa de monitoramento.

### Relatório

Ao encerrar o programa (tecla 'q' ou ESC), ele gera automaticamente um relatório com o histórico da contagem de gatos filhotes detectados ao longo da execução:

- Um gráfico em PNG (desenhado "na mão", só com funções do OpenCV, sem bibliotecas externas de plotagem), mostrando a quantidade de gatos ao longo do tempo em formato de degrau;
- Um CSV com os dados brutos (tempo em segundos x quantidade de gatos), para quem quiser analisar os dados em outra ferramenta (Excel, Python, etc.).

O objetivo é poder olhar depois como foi o comportamento dos gatos ao longo do dia/noite, quando eles saíam mais da caixa, etc.

## Estrutura do repositório

```
.
├── docs/
│   ├── contexto.txt        # anotações/contexto do projeto
│   └── main.cpp            # versão anterior/alternativa do código
├── exp_ncnn_model/
│   ├── metadata.yaml
│   ├── model.ncnn.bin       # pesos do modelo treinado
│   ├── model.ncnn.param     # arquitetura do modelo
│   └── model_ncnn.py
├── exp.pt                  # modelo original exportado pelo Ultralytics (PyTorch)
├── mainT.cpp                # código principal do programa
├── toque.sh                 # script de fallback para tocar notificação
├── gato_seg3_toque          # binário compilado
└── prints/                  # capturas de tela / fotos salvas pelo programa
```

## Dependências e como rodar

- OpenCV (versão 5, usa o módulo `geometry` para `contourArea`/`approxPolyDP`)
- NCNN, compilado com suporte a Vulkan
- Driver Vulkan para a GPU (ex: `vulkan-radeon` ou `amdvlk` para AMD no Linux)
- FFmpeg (para o OpenCV conseguir abrir o stream RTSP)
- Compilador C++ com suporte a C++17 ou superior

Antes de rodar, ajuste em `mainT.cpp` o endereço da câmera:

```cpp
static const char* RTSP_URL = "rtsp://usuario:senha@IP_LOCAL:554/onvif1";
```

## Controles durante a execução

| Tecla | Ação |
|---|---|
| `f` | Salva uma foto pura (sem marcações) da câmera |
| `q` ou `ESC` | Encerra o programa e gera o relatório (PNG + CSV) |

Na inicialização, o programa também pede para selecionar com o mouse a região da caixa a ser monitorada (ENTER/espaço confirma, ESC cancela o monitoramento de fuga).


## Créditos

- Modelo base: arquitetura YOLO-seg, da [Ultralytics](https://www.ultralytics.com/);
- Conversão para NCNN: [NCNN](https://github.com/Tencent/ncnn) e [PNNX](https://github.com/pnnx/pnnx) (Tencent);
- Dataset e treinamento: [próprios, feitos pelo autor deste projeto](https://platform.ultralytics.com/gabriel-ch/datasets/eye-on-cat-v1);.
