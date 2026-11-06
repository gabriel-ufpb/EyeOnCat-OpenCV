
// main.cpp
// Detecta e SEGMENTA gatos filhotes em uma câmera RTSP em tempo real.
// Inferência via NCNN acelerada por GPU (Vulkan) — funciona bem em GPUs AMD como a RX6650XT,
// já que não depende de ROCm/CUDA.
//
// Pipeline:
//   RTSP (UDP) --OpenCV/FFmpeg--> frame --letterbox--> NCNN (Vulkan) --> boxes + coefs de máscara
//   máscara = sigmoid(protótipos · coefs) --> binariza --> findContours --> polígono --> desenha
 
#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp> // OpenCV 5: contourArea/approxPolyDP foram para o modulo geometry
#include <net.h>
#include <gpu.h>
 
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
 
// ------------------------------------------------------------------
// CONFIGURAÇÕES — ajuste aqui
// ------------------------------------------------------------------
static const char* RTSP_URL   = "rtsp://admin:<SEBHA>@<IP_LOCAL>:554/onvif1";
static const char* PARAM_PATH = "exp_ncnn_model/model.ncnn.param";
static const char* BIN_PATH   = "exp_ncnn_model/model.ncnn.bin";
 
static const int   INPUT_SIZE  = 640;   // deve bater com o imgsz usado no export
static const int   NUM_CLASSES = 1;     // seu modelo tem 1 classe: "gato filhote"
static const int   MASK_DIM    = 32;    // padrão dos modelos *-seg do Ultralytics
static const float CONF_THRESH = 0.65f;
static const float NMS_THRESH  = 0.45f;
static const float MASK_THRESH = 0.5f;
 
static const char* CLASS_NAMES[NUM_CLASSES] = { "gato filhote" };
 
// Monitoramento da caixa: quantos gatos filhotes precisam estar dentro
// para o alerta de "fuga de gato" ser desligado.
static const int   TARGET_CATS_INSIDE = 6;
static const double BEEP_INTERVAL_SEC = 1.5; // intervalo entre bipes enquanto o alerta estiver ativo

// Rastreamento temporal (reduz alucinacoes de 1 frame só):
// uma deteccao só é contada/desenhada (mascara + contador na tela) depois
// de aparecer em MIN_HITS_TO_CONFIRM frames consecutivos (aprox.) na
// mesma posicao. NÃO afeta o alerta de fuga: esse é disparado na hora,
// com base na deteccao crua do frame atual (ver box_monitor.update()).
static const int   MIN_HITS_TO_CONFIRM = 3;    // frames seguidos até "confirmar" um gato
static const int   MAX_MISSES_TRACK    = 5;    // frames sem deteccao até desistir do rastro
static const float TRACK_IOU_THRESH    = 0.30f;// IoU minimo p/ casar deteccao com rastro existente
static const float TRACK_EMA_ALPHA     = 0.30f;// suavizacao da confianca (media movel exponencial)
 
// ------------------------------------------------------------------
struct Detection {
    cv::Rect2f box; // em coordenadas da imagem letterboxed (INPUT_SIZE x INPUT_SIZE)
    float conf;
    int cls;
    std::vector<float> mask_coeff;
};
 
static void letterbox(const cv::Mat& src, cv::Mat& dst, float& scale,
                       int& pad_x, int& pad_y, int& new_w, int& new_h) {
    int w = src.cols, h = src.rows;
    scale = std::min((float)INPUT_SIZE / w, (float)INPUT_SIZE / h);
    new_w = (int)std::round(w * scale);
    new_h = (int)std::round(h * scale);
 
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));
 
    dst = cv::Mat(INPUT_SIZE, INPUT_SIZE, CV_8UC3, cv::Scalar(114, 114, 114));
    pad_x = (INPUT_SIZE - new_w) / 2;
    pad_y = (INPUT_SIZE - new_h) / 2;
    resized.copyTo(dst(cv::Rect(pad_x, pad_y, new_w, new_h)));
}
 
static inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
 
static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni <= 0.f ? 0.f : inter / uni;
}
 
static std::vector<int> nms(const std::vector<Detection>& dets) {
    std::vector<int> idx(dets.size());
    for (size_t i = 0; i < idx.size(); i++) idx[i] = (int)i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return dets[a].conf > dets[b].conf; });
 
    std::vector<bool> removed(dets.size(), false);
    std::vector<int> keep;
    for (size_t i = 0; i < idx.size(); i++) {
        int cur = idx[i];
        if (removed[cur]) continue;
        keep.push_back(cur);
        for (size_t j = i + 1; j < idx.size(); j++) {
            int other = idx[j];
            if (removed[other]) continue;
            if (dets[cur].cls == dets[other].cls && iou(dets[cur].box, dets[other].box) > NMS_THRESH)
                removed[other] = true;
        }
    }
    return keep;
}
 
// ------------------------------------------------------------------
// RELATÓRIO — histórico da contagem de gatos filhotes ao longo do tempo
// ------------------------------------------------------------------
struct CountSample {
    double t_sec; // segundos desde o início da execução
    int count;
};
 
static std::string timestamp_string() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    localtime_r(&t, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y%m%d_%H%M%S");
    return oss.str();
}
 
// Desenha um gráfico de linha simples (sem dependências externas, só OpenCV)
// e salva como PNG. Também salva os dados brutos em CSV.
static void save_report(const std::vector<CountSample>& history) {
    if (history.empty()) {
        printf("Sem dados suficientes para gerar o relatorio.\n");
        return;
    }
 
    std::string stamp    = timestamp_string();
    std::string png_path = "relatorio_gatos_" + stamp + ".png";
    std::string csv_path = "relatorio_gatos_" + stamp + ".csv";
 
    // ---- CSV com os dados brutos ----
    FILE* csv = fopen(csv_path.c_str(), "w");
    if (csv) {
        fprintf(csv, "tempo_s,quantidade_gatos\n");
        for (auto& s : history) fprintf(csv, "%.2f,%d\n", s.t_sec, s.count);
        fclose(csv);
    } else {
        fprintf(stderr, "Aviso: nao foi possivel escrever %s\n", csv_path.c_str());
    }
 
    // ---- Limites dos eixos ----
    double t_max = history.back().t_sec;
    if (t_max <= 0.0) t_max = 1.0;
    int count_max = 0, count_min = history[0].count;
    for (auto& s : history) {
        count_max = std::max(count_max, s.count);
        count_min = std::min(count_min, s.count);
    }
    count_max = std::max(count_max, count_min + 1); // garante um eixo Y com faixa != 0
 
    // ---- Canvas ----
    const int W = 1200, H = 600;
    const int margin_l = 60, margin_r = 30, margin_t = 60, margin_b = 55;
    const int plot_w = W - margin_l - margin_r;
    const int plot_h = H - margin_t - margin_b;
    cv::Mat chart(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
 
    auto to_px = [&](double t, int c) -> cv::Point {
        double x_frac = t / t_max;
        double y_frac = (double)(c - count_min) / (count_max - count_min);
        int x = margin_l + (int)std::round(x_frac * plot_w);
        int y = margin_t + plot_h - (int)std::round(y_frac * plot_h);
        return cv::Point(x, y);
    };
 
    // Eixos
    cv::line(chart, cv::Point(margin_l, margin_t), cv::Point(margin_l, margin_t + plot_h), cv::Scalar(0, 0, 0), 2);
    cv::line(chart, cv::Point(margin_l, margin_t + plot_h), cv::Point(margin_l + plot_w, margin_t + plot_h), cv::Scalar(0, 0, 0), 2);
 
    // Grade horizontal + rótulos do eixo Y (valores inteiros de contagem)
    for (int c = count_min; c <= count_max; c++) {
        cv::Point p = to_px(0, c);
        cv::line(chart, cv::Point(margin_l, p.y), cv::Point(margin_l + plot_w, p.y), cv::Scalar(230, 230, 230), 1);
        cv::putText(chart, std::to_string(c), cv::Point(10, p.y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
 
    // Marcações do eixo X (tempo)
    const int n_ticks = 6;
    for (int i = 0; i <= n_ticks; i++) {
        double t = t_max * i / n_ticks;
        cv::Point p = to_px(t, count_min);
        char buf[32];
        snprintf(buf, sizeof(buf), "%.0fs", t);
        cv::putText(chart, buf, cv::Point(p.x - 15, margin_t + plot_h + 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        cv::line(chart, cv::Point(p.x, margin_t + plot_h), cv::Point(p.x, margin_t + plot_h + 5), cv::Scalar(0, 0, 0), 1);
    }
 
    // Linha do gráfico (formato "degrau", já que a contagem é um valor inteiro que muda de vez em quando)
    for (size_t i = 1; i < history.size(); i++) {
        cv::Point p1 = to_px(history[i - 1].t_sec, history[i - 1].count);
        cv::Point p2 = to_px(history[i].t_sec, history[i - 1].count); // segmento horizontal
        cv::Point p3 = to_px(history[i].t_sec, history[i].count);     // segmento vertical (degrau)
        cv::line(chart, p1, p2, cv::Scalar(0, 140, 255), 2, cv::LINE_AA);
        cv::line(chart, p2, p3, cv::Scalar(0, 140, 255), 2, cv::LINE_AA);
    }
 
    cv::putText(chart, "Gatos filhotes detectados ao longo do tempo", cv::Point(margin_l, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    cv::putText(chart, "Tempo (s)", cv::Point(W / 2 - 40, H - 12), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 1);
    cv::putText(chart, "Qtd. gatos", cv::Point(margin_l, margin_t - 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
 
    cv::imwrite(png_path, chart);
    printf("Relatorio (grafico) salvo em: %s\n", png_path.c_str());
    printf("Dados brutos (CSV) salvos em: %s\n", csv_path.c_str());
}
 
// ------------------------------------------------------------------
// CatBoxMonitor — seleciona uma caixa na imagem, conta quantos gatos
// filhotes estao dentro/fora dela, e dispara um alerta visual + sonoro
// de "fuga de gato" sempre que houver gato fora e a meta de gatos
// dentro ainda nao tiver sido atingida.
// ------------------------------------------------------------------
class CatBoxMonitor {
public:
    CatBoxMonitor(int target_inside_count, double beep_interval_sec = 1.5)
        : target_inside_(target_inside_count), beep_interval_(beep_interval_sec) {}
 
    ~CatBoxMonitor() { stop_alert_thread(); }
 
    // Deixa o usuario desenhar a caixa na janela informada antes do loop principal.
    // ENTER/SPACE confirma a selecao, ESC cancela (monitoramento fica desativado).
    void select_box(const cv::Mat& frame, const std::string& window_name) {
        printf("Selecione a caixa dos gatos filhotes na janela e pressione ENTER/SPACE "
               "(ESC cancela).\n");
        cv::Rect r = cv::selectROI(window_name, frame, false, false);
        if (r.width > 0 && r.height > 0) {
            box_ = r;
            has_box_ = true;
            printf("Caixa selecionada: x=%d y=%d w=%d h=%d\n", r.x, r.y, r.width, r.height);
        } else {
            fprintf(stderr, "Nenhuma caixa selecionada - monitoramento de fuga desativado.\n");
            has_box_ = false;
        }
    }
 
    bool has_box() const { return has_box_; }
 
    // Recebe os centros (coords da imagem original) de cada gato detectado no frame atual.
    void update(const std::vector<cv::Point>& cat_centers) {
        int inside = 0, outside = 0;
        if (has_box_) {
            for (const auto& c : cat_centers) {
                if (box_.contains(c)) inside++;
                else outside++;
            }
        } else {
            outside = (int)cat_centers.size();
        }
        inside_count_  = inside;
        outside_count_ = outside;
 
        bool should_alert = has_box_ && outside_count_ > 0 && inside_count_ < target_inside_;
 
        if (should_alert && !alert_active_.load()) {
            start_alert();
        } else if (!should_alert && alert_active_.load()) {
            stop_alert();
        }
    }
 
    int  inside_count()  const { return inside_count_; }
    int  outside_count() const { return outside_count_; }
    bool alert_active()  const { return alert_active_.load(); }
 
    // Desenha a caixa, contadores e o banner de alerta (se ativo) no frame.
    void draw(cv::Mat& frame) const {
        if (has_box_) {
            cv::Scalar box_color = alert_active_.load() ? cv::Scalar(0, 0, 255)
                                                          : cv::Scalar(255, 200, 0);
            cv::rectangle(frame, box_, box_color, 2);
            cv::putText(frame, "Caixa", cv::Point(box_.x, std::max(box_.y - 8, 12)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv::LINE_AA);
        }
 
        char info[96];
        snprintf(info, sizeof(info), "Dentro: %d  Fora: %d  (meta: %d)",
                  inside_count_, outside_count_, target_inside_);
        cv::putText(frame, info, cv::Point(15, frame.rows - 15),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
 
        if (alert_active_.load()) {
            // Pisca a cada ~500ms.
            bool blink_on = (std::chrono::duration_cast<std::chrono::milliseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch()).count()
                             / 500) % 2 == 0;
            if (blink_on) {
                const std::string text = "FUGA DE GATO!";
                int baseline = 0;
                double scale = 1.4;
                cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 3, &baseline);
                int x = (frame.cols - ts.width) / 2;
                int y = 60;
                cv::rectangle(frame, cv::Point(x - 20, y - ts.height - 15),
                              cv::Point(x + ts.width + 20, y + baseline + 10),
                              cv::Scalar(0, 0, 200), cv::FILLED);
                cv::putText(frame, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale,
                            cv::Scalar(255, 255, 255), 3, cv::LINE_AA);
            }
        }
    }
 
private:
    void start_alert() {
        alert_active_.store(true);
        stop_thread_.store(false);
        sound_thread_ = std::thread([this] {
            while (!stop_thread_.load()) {
                play_beep();
                std::this_thread::sleep_for(std::chrono::duration<double>(beep_interval_));
            }
        });
        printf("[ALERTA] Fuga de gato detectada! Gatos fora da caixa: %d\n", outside_count_);
        play_ring(); // dispara uma unica vez por ativacao do alerta
    }
 
    void stop_alert() {
        alert_active_.store(false);
        stop_alert_thread();
        printf("[OK] Meta atingida (%d gatos dentro da caixa). Alerta encerrado.\n", target_inside_);
    }
 
    void stop_alert_thread() {
        stop_thread_.store(true);
        if (sound_thread_.joinable()) sound_thread_.join();
    }
 
    // Dispara um beep curto sem travar o loop principal (comando roda em background).
    // Usa alsa-utils (speaker-test), que normalmente ja vem instalado no Arch.
    // Troque por outro comando (ex: "paplay /caminho/som.wav &") se preferir um som customizado.
    static void play_beep() {
        std::system("(speaker-test -t sine -f 1000 -l 1 > /dev/null 2>&1 &) ");
    }

    // Dispara UMA VEZ, ao inicio de cada ativacao do alerta, um comando
    // externo para chamar atencao (ex: fazer o celular tocar via
    // GSConnect). Se o gdbus falhar (GSConnect nao instalado/configurado,
    // dispositivo offline, etc.), cai automaticamente para rodar o script
    // "toque.sh", que deve estar no mesmo diretorio de trabalho do
    // programa (mesmo diretorio do main.cpp / do binario compilado).
    // Roda em background para nao travar o loop principal.
    static void play_ring() {
        std::system(
            "(gdbus call --session --dest org.gnome.Shell.Extensions.GSConnect "
            "--object-path /org/gnome/Shell/Extensions/GSConnect/Device/d73a7ac26db841679592534886839476 "
            "--method org.gtk.Actions.Activate \"ring\" [] {} > /dev/null 2>&1 "
            "|| ./toque.sh > /dev/null 2>&1) &");
    }
 
    cv::Rect box_;
    bool     has_box_ = false;
 
    int      target_inside_;
    double   beep_interval_;
 
    int      inside_count_  = 0;
    int      outside_count_ = 0;
 
    std::atomic<bool> alert_active_{false};
    std::atomic<bool> stop_thread_{false};
    std::thread        sound_thread_;
};

// ------------------------------------------------------------------
// SnapshotSaver — salva o frame "limpo" (sem caixas, mascaras, texto
// ou qualquer outra marcacao desenhada por cima) sempre que o usuario
// aciona uma tecla. O arquivo PNG é salvo no diretorio de trabalho
// atual (o diretorio de onde o programa foi executado), com um nome
// unico baseado em data/hora.
// ------------------------------------------------------------------
class SnapshotSaver {
public:
    explicit SnapshotSaver(std::string prefix = "foto") : prefix_(std::move(prefix)) {}

    // Recebe o frame cru (deve ser uma copia feita ANTES de qualquer
    // cv::rectangle/putText/polylines/addWeighted etc.) e grava em disco.
    void save(const cv::Mat& clean_frame) {
        if (clean_frame.empty()) {
            fprintf(stderr, "Aviso: tentativa de salvar foto com frame vazio.\n");
            return;
        }
        std::string path = prefix_ + "_" + timestamp_string() + ".png";
        if (cv::imwrite(path, clean_frame)) {
            printf("[FOTO] Salva em: %s\n", path.c_str());
        } else {
            fprintf(stderr, "Erro: nao foi possivel salvar a foto em %s\n", path.c_str());
        }
    }

private:
    std::string prefix_;
};

// ------------------------------------------------------------------
// TemporalTracker — "acumula contexto" entre frames.
// Em vez de confiar em cada frame isoladamente, cada deteccao vira um
// "rastro" (track). Um rastro só é considerado um gato de verdade
// (confirmed) depois de aparecer em MIN_HITS_TO_CONFIRM frames
// seguidos, aproximadamente na mesma posicao (casado por IoU).
// A confianca exibida é uma media movel exponencial (EMA), entao um
// pico isolado de 95% em cima de uma sombra não é suficiente sozinho.
// Isso elimina a maioria dos falsos positivos de 1 frame sem precisar
// mexer no threshold de confianca (que, como voce notou, cortaria
// acertos junto com os erros).
// ------------------------------------------------------------------
struct Track {
    int id = -1;
    cv::Rect2f box;                 // ultima caixa conhecida (coords da imagem original)
    float conf_ema = 0.f;           // confianca suavizada (EMA)
    int hits = 0;                   // frames consecutivos com deteccao casada
    int misses = 0;                 // frames consecutivos sem deteccao casada
    bool confirmed = false;
    std::vector<cv::Point> poly;    // ultimo poligono valido (para desenhar mesmo em "miss" curto)
};

class TemporalTracker {
public:
    TemporalTracker(int min_hits_to_confirm, int max_misses,
                     float iou_match_thresh, float ema_alpha)
        : min_hits_(min_hits_to_confirm), max_misses_(max_misses),
          iou_thresh_(iou_match_thresh), alpha_(ema_alpha) {}

    // Recebe as deteccoes cruas do frame atual (coords da imagem original) e
    // devolve os indices, dentro de tracks(), que estao CONFIRMADOS agora.
    std::vector<int> update(const std::vector<cv::Rect2f>& det_boxes,
                             const std::vector<float>& det_confs,
                             const std::vector<std::vector<cv::Point>>& det_polys) {
        std::vector<bool> det_used(det_boxes.size(), false);

        // Casa cada rastro existente com a deteccao de maior IoU no frame atual
        for (auto& tr : tracks_) {
            int best_d = -1;
            float best_iou = iou_thresh_;
            for (size_t d = 0; d < det_boxes.size(); d++) {
                if (det_used[d]) continue;
                float v = iou(tr.box, det_boxes[d]);
                if (v > best_iou) { best_iou = v; best_d = (int)d; }
            }
            if (best_d >= 0) {
                tr.box  = det_boxes[best_d];
                tr.poly = det_polys[best_d];
                tr.conf_ema = (tr.hits == 0) ? det_confs[best_d]
                                              : alpha_ * det_confs[best_d] + (1.f - alpha_) * tr.conf_ema;
                tr.hits++;
                tr.misses = 0;
                if (tr.hits >= min_hits_) tr.confirmed = true;
                det_used[best_d] = true;
            } else {
                tr.misses++;
            }
        }

        // Deteccoes sem rastro correspondente comecam um rastro novo (ainda nao confirmado)
        for (size_t d = 0; d < det_boxes.size(); d++) {
            if (det_used[d]) continue;
            Track tr;
            tr.id = next_id_++;
            tr.box = det_boxes[d];
            tr.poly = det_polys[d];
            tr.conf_ema = det_confs[d];
            tr.hits = 1;
            tracks_.push_back(std::move(tr));
        }

        // Remove rastros "mortos" (muitos frames seguidos sem casar com nada)
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                      [&](const Track& tr) { return tr.misses > max_misses_; }),
                      tracks_.end());

        std::vector<int> confirmed_idx;
        for (size_t t = 0; t < tracks_.size(); t++)
            if (tracks_[t].confirmed) confirmed_idx.push_back((int)t);
        return confirmed_idx;
    }

    const std::vector<Track>& tracks() const { return tracks_; }

private:
    static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
        float inter = (a & b).area();
        float uni = a.area() + b.area() - inter;
        return uni <= 0.f ? 0.f : inter / uni;
    }

    std::vector<Track> tracks_;
    int   next_id_ = 0;
    int   min_hits_;
    int   max_misses_;
    float iou_thresh_;
    float alpha_;
};
 
int main() {
    // ---------------------------------------------------------------
    // 1) Abre o stream RTSP forçando transporte UDP
    // ---------------------------------------------------------------
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;udp", 1);
    cv::VideoCapture cap(RTSP_URL, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        fprintf(stderr, "Erro: nao foi possivel abrir o stream RTSP: %s\n", RTSP_URL);
        return 1;
    }
 
    // ---------------------------------------------------------------
    // 2) Inicializa GPU Vulkan e carrega o modelo NCNN
    // ---------------------------------------------------------------
    ncnn::create_gpu_instance();
    int gpu_count = ncnn::get_gpu_count();
    printf("GPUs Vulkan detectadas: %d\n", gpu_count);
    if (gpu_count == 0) {
        fprintf(stderr, "Aviso: nenhuma GPU Vulkan encontrada. Verifique o driver "
                         "(vulkan-radeon / amdvlk) e 'vulkaninfo'. Rodando em CPU.\n");
    }
 
    ncnn::Net net;
    net.opt.use_vulkan_compute   = gpu_count > 0;
    net.opt.use_fp16_packed      = true;
    net.opt.use_fp16_storage     = true;
    net.opt.use_fp16_arithmetic  = true;
    net.opt.use_packing_layout   = true;
    net.opt.num_threads          = (int)std::thread::hardware_concurrency();
 
    if (net.load_param(PARAM_PATH) != 0) {
        fprintf(stderr, "Erro ao carregar %s\n", PARAM_PATH);
        return 1;
    }
    if (net.load_model(BIN_PATH) != 0) {
        fprintf(stderr, "Erro ao carregar %s\n", BIN_PATH);
        return 1;
    }
 
    std::vector<const char*> in_names  = net.input_names();
    std::vector<const char*> out_names = net.output_names();
    if (in_names.empty() || out_names.size() < 2) {
        fprintf(stderr, "Modelo com formato inesperado (esperava 1 entrada e 2 saidas).\n"
                         "Confira os nomes de blob no arquivo .param.\n");
        return 1;
    }
    const char* in_name   = in_names[0];
    const char* out0_name = out_names[0]; // detecções: boxes + scores + coefs de máscara
    const char* out1_name = out_names[1]; // protótipos de máscara
 
    printf("Blob de entrada: %s\n", in_name);
    printf("Blobs de saida : %s , %s\n", out0_name, out1_name);
    printf("Vulkan (GPU) ativo: %s\n", net.opt.use_vulkan_compute ? "sim" : "nao");
 
    cv::namedWindow("Gatos Filhotes - Segmentacao", cv::WINDOW_NORMAL);
 
    // ---------------------------------------------------------------
    // 2.5) Captura um frame inicial e deixa o usuario selecionar a
    //      caixa que sera monitorada (contagem dentro/fora + alerta)
    // ---------------------------------------------------------------
    CatBoxMonitor box_monitor(TARGET_CATS_INSIDE, BEEP_INTERVAL_SEC);
    {
        cv::Mat calib_frame;
        if (cap.read(calib_frame) && !calib_frame.empty()) {
            box_monitor.select_box(calib_frame, "Gatos Filhotes - Segmentacao");
        } else {
            fprintf(stderr, "Aviso: nao foi possivel capturar frame inicial para selecionar "
                             "a caixa. Monitoramento de fuga desativado.\n");
        }
    }
 
    bool printed_shapes = false;
    cv::Mat frame;
 
    std::vector<CountSample> history;
    auto run_start = std::chrono::steady_clock::now();

    // Acumula contexto entre frames para so contar/desenhar gatos que
    // se confirmam ao longo do tempo (ver classe TemporalTracker acima).
    TemporalTracker tracker(MIN_HITS_TO_CONFIRM, MAX_MISSES_TRACK,
                             TRACK_IOU_THRESH, TRACK_EMA_ALPHA);

    // Salva fotos "puras" (sem marcacoes) da camera quando o usuario
    // aperta a tecla 'f' com a janela em foco.
    SnapshotSaver snapshot_saver;
    printf("Dica: pressione 'f' na janela do video para salvar uma foto pura (sem marcacoes).\n");
 
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            fprintf(stderr, "Frame vazio, tentando reconectar...\n");
            cap.release();
            std::this_thread::sleep_for(std::chrono::milliseconds(800));
            cap.open(RTSP_URL, cv::CAP_FFMPEG);
            continue;
        }

        // Copia do frame cru, antes de qualquer desenho/marcacao,
        // para o botao de "tirar foto" salvar a imagem pura da camera.
        cv::Mat clean_frame = frame.clone();
 
        // -------------------------------------------------------
        // 3) Pré-processamento: letterbox + BGR->RGB + normaliza
        // -------------------------------------------------------
        float scale;
        int pad_x, pad_y, new_w, new_h;
        cv::Mat letterboxed;
        letterbox(frame, letterboxed, scale, pad_x, pad_y, new_w, new_h);
 
        cv::Mat rgb;
        cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
 
        ncnn::Mat in = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB, INPUT_SIZE, INPUT_SIZE);
        const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
        in.substract_mean_normalize(nullptr, norm_vals);
 
        // -------------------------------------------------------
        // 4) Inferência
        // -------------------------------------------------------
        ncnn::Extractor ex = net.create_extractor();
        ex.input(in_name, in);
 
        ncnn::Mat out0, out1;
        ex.extract(out0_name, out0);
        ex.extract(out1_name, out1);
 
        if (!printed_shapes) {
            printf("out0: w=%d h=%d c=%d | out1: w=%d h=%d c=%d\n",
                   out0.w, out0.h, out0.c, out1.w, out1.h, out1.c);
            printf("(esperado out0: w=num_predicoes, h=%d ; out1: w/h=tamanho do proto, c=%d)\n",
                   4 + NUM_CLASSES + MASK_DIM, MASK_DIM);
            printed_shapes = true;
        }
 
        int num_preds = out0.w;
 
        // -------------------------------------------------------
        // 5) Decodifica caixas + scores + coeficientes de máscara
        // -------------------------------------------------------
        std::vector<Detection> candidates;
        candidates.reserve(64);
 
        for (int i = 0; i < num_preds; i++) {
            float cx = out0.row(0)[i];
            float cy = out0.row(1)[i];
            float w  = out0.row(2)[i];
            float h  = out0.row(3)[i];
 
            int best_cls = 0;
            float best_score = out0.row(4)[i];
            for (int c = 1; c < NUM_CLASSES; c++) {
                float s = out0.row(4 + c)[i];
                if (s > best_score) { best_score = s; best_cls = c; }
            }
            if (best_score < CONF_THRESH) continue;
 
            Detection det;
            det.box  = cv::Rect2f(cx - w / 2.f, cy - h / 2.f, w, h);
            det.conf = best_score;
            det.cls  = best_cls;
            det.mask_coeff.resize(MASK_DIM);
            for (int m = 0; m < MASK_DIM; m++)
                det.mask_coeff[m] = out0.row(4 + NUM_CLASSES + m)[i];
 
            candidates.push_back(std::move(det));
        }
 
        std::vector<int> keep = nms(candidates);
 
        int proto_w = out1.w, proto_h = out1.h;
 
        cv::Mat overlay = frame.clone();
        cv::Scalar color(0, 255, 0);

        // ---- 1) Extrai poligono + bbox (coords originais) de cada deteccao crua ----
        std::vector<cv::Rect2f> det_boxes_orig;
        std::vector<float> det_confs;
        std::vector<std::vector<cv::Point>> det_polys;

        for (int idx : keep) {
            const Detection& det = candidates[idx];

            // ---- Reconstrói a máscara a partir dos protótipos ----
            cv::Mat mask_small(proto_h, proto_w, CV_32FC1, cv::Scalar(0));
            for (int m = 0; m < MASK_DIM; m++) {
                const float* proto_plane = out1.channel(m);
                float coeff = det.mask_coeff[m];
                if (coeff == 0.f) continue;
                for (int y = 0; y < proto_h; y++) {
                    float* row = mask_small.ptr<float>(y);
                    const float* prow = proto_plane + y * proto_w;
                    for (int x = 0; x < proto_w; x++)
                        row[x] += coeff * prow[x];
                }
            }
            mask_small.forEach<float>([](float& v, const int*) { v = sigmoid(v); });

            cv::Mat mask_full;
            cv::resize(mask_small, mask_full, cv::Size(INPUT_SIZE, INPUT_SIZE), 0, 0, cv::INTER_LINEAR);

            cv::Rect valid_region(pad_x, pad_y, new_w, new_h);
            valid_region &= cv::Rect(0, 0, mask_full.cols, mask_full.rows);
            if (valid_region.width <= 0 || valid_region.height <= 0) continue;

            cv::Mat mask_orig;
            cv::resize(mask_full(valid_region), mask_orig, frame.size(), 0, 0, cv::INTER_LINEAR);

            // ---- Bbox de volta para coordenadas da imagem original ----
            cv::Rect2f box_orig(
                (det.box.x - pad_x) / scale,
                (det.box.y - pad_y) / scale,
                det.box.width / scale,
                det.box.height / scale);

            int bx  = std::max(0, (int)std::round(box_orig.x));
            int by  = std::max(0, (int)std::round(box_orig.y));
            int bx2 = std::min(frame.cols, (int)std::round(box_orig.x + box_orig.width));
            int by2 = std::min(frame.rows, (int)std::round(box_orig.y + box_orig.height));
            if (bx2 <= bx || by2 <= by) continue;
            cv::Rect box_i(bx, by, bx2 - bx, by2 - by);

            // ---- Binariza e restringe a máscara à área do bbox ----
            cv::Mat mask_bin;
            cv::threshold(mask_orig, mask_bin, MASK_THRESH, 255, cv::THRESH_BINARY);
            mask_bin.convertTo(mask_bin, CV_8UC1);

            cv::Mat mask_roi = cv::Mat::zeros(mask_bin.size(), CV_8UC1);
            mask_bin(box_i).copyTo(mask_roi(box_i));

            // ---- Extrai o(s) polígono(s) do contorno da máscara ----
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask_roi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            // Fica com o maior contorno valido (o "gato" principal da deteccao)
            std::vector<cv::Point> best_poly;
            double best_area = 0.0;
            for (auto& contour : contours) {
                double area = cv::contourArea(contour);
                if (area < 60) continue; // ignora ruído
                std::vector<cv::Point> poly;
                cv::approxPolyDP(contour, poly, 2.0, true);
                if (poly.size() < 3) continue;
                if (area > best_area) { best_area = area; best_poly = std::move(poly); }
            }
            if (best_poly.empty()) continue;

            det_boxes_orig.push_back(box_i);
            det_confs.push_back(det.conf);
            det_polys.push_back(std::move(best_poly));
        }

        // ---- Centros das deteccoes CRUAS deste frame (sem esperar confirmacao) ----
        // Usados só para o alerta de "fuga de gato": basta 1 gato aparecer fora
        // da caixa em um único frame para disparar, sem tempo de espera.
        std::vector<cv::Point> raw_cat_centers;
        raw_cat_centers.reserve(det_boxes_orig.size());
        for (const auto& b : det_boxes_orig)
            raw_cat_centers.push_back(cv::Point((int)(b.x + b.width / 2), (int)(b.y + b.height / 2)));

        // ---- 2) Acumula contexto: casa com rastros de frames anteriores e só ----
        //         considera "confirmado" o que se repetiu por MIN_HITS_TO_CONFIRM frames.
        //         (Isso ainda é usado para desenhar/contar os gatos na tela, mas
        //         NÃO é mais usado para o alerta de fuga, que agora é instantâneo.)
        std::vector<int> confirmed_idx = tracker.update(det_boxes_orig, det_confs, det_polys);

        // ---- 3) Desenha e conta apenas as tracks confirmadas ----
        int count = 0;
        const std::vector<Track>& tracks = tracker.tracks();

        for (int t : confirmed_idx) {
            const Track& tr = tracks[t];

            cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{tr.poly}, color);
            cv::polylines(frame, std::vector<std::vector<cv::Point>>{tr.poly}, true, color, 2, cv::LINE_AA);

            count++;

            char label[96];
            snprintf(label, sizeof(label), "%s %.0f%%", CLASS_NAMES[0], tr.conf_ema * 100);
            cv::putText(frame, label, cv::Point((int)tr.box.x, std::max((int)tr.box.y - 8, 12)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
        }

        // ---- Registra amostra para o relatório final ----
        // Grava sempre que a contagem muda (transição fica bem marcada no gráfico)
        // e também a cada ~1s mesmo sem mudança, para servir de "heartbeat".
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
        if (history.empty() || count != history.back().count || elapsed - history.back().t_sec >= 1.0) {
            history.push_back({ elapsed, count });
        }
 
        cv::addWeighted(overlay, 0.35, frame, 0.65, 0, frame);
 
        // ---- Atualiza contagem dentro/fora da caixa e alerta de fuga ----
        // Usa raw_cat_centers (deteccoes do frame atual, sem confirmacao
        // por múltiplos frames) para que a fuga seja detectada na hora.
        box_monitor.update(raw_cat_centers);
        box_monitor.draw(frame);
 
        // ---- Contador no topo da tela ----
        char counter_text[64];
        snprintf(counter_text, sizeof(counter_text), "Gatos filhotes: %d", count);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(counter_text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
        cv::rectangle(frame, cv::Point(10, 10), cv::Point(20 + ts.width, 20 + ts.height + baseline),
                      cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, counter_text, cv::Point(15, 20 + ts.height),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
 
        cv::imshow("Gatos Filhotes - Segmentacao", frame);
        int key = cv::waitKey(1);
        if (key == 'f' || key == 'F') {
            snapshot_saver.save(clean_frame);
        }
        if (key == 'q' || key == 27) break;
    }
 
    // garante que o último valor de contagem apareça no gráfico
    {
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - run_start).count();
        if (history.empty() || history.back().t_sec < elapsed)
            history.push_back({ elapsed, history.empty() ? 0 : history.back().count });
    }
    save_report(history);
 
    cap.release();
    cv::destroyAllWindows();
    ncnn::destroy_gpu_instance();
    return 0;
}
