/* Theme (light/dark) + language (vi/en) toggle -- shared across all pages. */
(function () {
  "use strict";
  const STORAGE_THEME = "vd_theme";
  const STORAGE_LANG = "vd_lang";

  const I18N = {
    vi: {
      nav_upload: "Tải lên",
      nav_camera: "Camera",
      nav_history: "Lịch sử",
      footer_text: "Phát hiện bạo lực trong video · PyTorch CNN–LSTM · localhost demo",
      theme_dark: "Tối",
      theme_light: "Sáng",

      camera_eyebrow: "GIÁM SÁT TRỰC TIẾP",
      camera_hero_h1: "Giám sát qua camera",
      camera_hero_p: "Phân tích trực tiếp và hiển thị xác suất Bạo lực / Không bạo lực.",
      camera_stat_status_label: "TRẠNG THÁI",
      camera_stat_status_value: "Thời gian thực",
      camera_stat_latency_label: "ĐỘ TRỄ",
      camera_stat_latency_value: "~1.1 giây",
      camera_stat_record_label: "GHI LỊCH SỬ",
      camera_stat_record_value: "Khi phát hiện",

      eyebrow_analytics: "PHÂN TÍCH VIDEO",
      upload_h1: "Phân tích video thông minh",
      upload_p: "Tải video lên để kiểm tra hành vi bạo lực bằng mô hình CNN–LSTM.",
      stat_model_label: "MÔ HÌNH",
      stat_input_label: "ĐẦU VÀO",
      stat_input_value: "Video / Webcam",
      stat_result_label: "KẾT QUẢ",
      stat_result_value: "Lưu vào lịch sử",
      section_check_h2: "Kiểm tra video",
      section_check_p: "Chọn tệp MP4, AVI, MOV hoặc kéo thả vào khung bên dưới.",
      status_ready: "Sẵn sàng",
      choose_video_title: "Chọn video",
      choose_video_hint: "Nhấn để duyệt tệp từ máy tính",
      btn_detect: "Phát hiện bạo lực",
      btn_reset: "Làm mới",
      alert_violence: "⚠ CẢNH BÁO BẠO LỰC ĐÃ PHÁT HIỆN",
      no_result_yet: "Chưa có kết quả",
      ready_to_detect: "Sẵn sàng nhận diện",
      detecting: "Đang nhận diện...",
      choose_video_first: "Chọn video trước",
      result_label: "Kết quả: ",
      confidence_label: "Độ tin cậy: ",
      saved_history_prefix: "Đã lưu lịch sử #",
      attention_label: "Attention (frame quyết định)",
      view_saved_video: "Xem video đã lưu",
      download_result: "Tải kết quả",
      error_label: "Lỗi: ",
      pred_violence: "Bạo lực",
      pred_nonviolence: "Không bạo lực",

      camera_h1: "Camera trực tiếp",
      camera_p: "Nhấn bật camera, chờ vài giây để mô hình tích lũy đủ khung hình.",
      btn_start_camera: "Bật camera",
      btn_stop_camera: "Tắt",
      btn_record_10s: "Quay thử 10s",
      camera_off: "Chưa bật camera",
      hud_default: "Bạo lực: —% · Không bạo lực: —%",
      camera_open_error: "Không mở được camera: ",
      camera_running: "Camera đang chạy · chờ vài giây để có kết quả",
      detecting_realtime: "Đang nhận diện realtime...",
      detect_error_generic: "Lỗi nhận diện",
      camera_stopped: "Đã tắt camera",
      record10_recording: "Đang quay thử: còn ",
      record10_uploading: "Đang gửi video cho model nhận diện...",
      record10_too_short: "Video quay được quá ngắn, thử lại.",
      overlay_violence: "Bạo lực",
      overlay_nonviolence: "Không bạo lực",

      history_h1: "Lịch sử nhận diện",
      history_p: "Ngày giờ · nguồn · nhãn · độ tin cậy · xem lại & tải video đã lưu",
      history_empty: "Chưa có bản ghi. Hãy upload hoặc bật camera trước.",
      history_count_suffix: " bản ghi gần nhất",
      btn_delete_all: "Xóa toàn bộ lịch sử",
      th_id: "ID",
      th_datetime: "Ngày giờ",
      th_source: "Nguồn",
      th_label: "Nhãn",
      th_confidence: "Độ tin cậy",
      th_video: "Video",
      th_actions: "Thao tác",
      pill_violence: "Bạo lực",
      pill_ok: "Không bạo lực",
      no_video: "Không có video",
      btn_view: "Xem",
      btn_download: "Tải xuống",
      btn_delete: "Xóa",
      confirm_delete_one: "Xóa bản ghi #{id} và video đã lưu?",
      confirm_delete_all: "Xóa toàn bộ lịch sử và tất cả video đã lưu?",
      err_delete_one: "Không thể xóa bản ghi.",
      err_delete_all: "Không thể xóa lịch sử.",
      source_upload: "Tải lên",
      source_webcam: "Webcam",
      source_rtsp: "RTSP",
    },
    en: {
      nav_upload: "Upload",
      nav_camera: "Camera",
      nav_history: "History",
      footer_text: "Violence detection in video · PyTorch CNN–LSTM · localhost demo",
      theme_dark: "Dark",
      theme_light: "Light",

      camera_eyebrow: "REALTIME MONITORING",
      camera_hero_h1: "Live camera monitoring",
      camera_hero_p: "Realtime analysis showing Violence / Non-violence probabilities.",
      camera_stat_status_label: "STATUS",
      camera_stat_status_value: "Realtime",
      camera_stat_latency_label: "LATENCY",
      camera_stat_latency_value: "~1.1 sec",
      camera_stat_record_label: "LOGGED",
      camera_stat_record_value: "On detection",

      eyebrow_analytics: "VIDEO ANALYTICS",
      upload_h1: "Smart Video Analysis",
      upload_p: "Upload a video to check for violent behavior using the CNN–LSTM model.",
      stat_model_label: "MODEL",
      stat_input_label: "INPUT",
      stat_input_value: "Video / Webcam",
      stat_result_label: "RESULT",
      stat_result_value: "Saved to history",
      section_check_h2: "Check a video",
      section_check_p: "Choose an MP4, AVI, MOV file, or drop one in the box below.",
      status_ready: "Ready",
      choose_video_title: "Choose video",
      choose_video_hint: "Click to browse a file from your computer",
      btn_detect: "Detect Violence",
      btn_reset: "Reset",
      alert_violence: "⚠ VIOLENCE ALERT DETECTED",
      no_result_yet: "No result yet",
      ready_to_detect: "Ready to detect",
      detecting: "Detecting...",
      choose_video_first: "Please choose a video first",
      result_label: "Result: ",
      confidence_label: "Confidence: ",
      saved_history_prefix: "Saved to history #",
      attention_label: "Attention (deciding frame)",
      view_saved_video: "View saved video",
      download_result: "Download result",
      error_label: "Error: ",
      pred_violence: "Violence",
      pred_nonviolence: "Non-violence",

      camera_h1: "Live Camera",
      camera_p: "Click start camera, wait a few seconds for the model to buffer enough frames.",
      btn_start_camera: "Start camera",
      btn_stop_camera: "Stop",
      btn_record_10s: "Record 10s test",
      camera_off: "Camera is off",
      hud_default: "Violence: —% · Non-violence: —%",
      camera_open_error: "Could not open camera: ",
      camera_running: "Camera running · wait a few seconds for results",
      detecting_realtime: "Detecting in realtime...",
      detect_error_generic: "Detection error",
      camera_stopped: "Camera stopped",
      record10_recording: "Recording test: ",
      record10_uploading: "Sending video to the model...",
      record10_too_short: "Recorded clip too short, try again.",
      overlay_violence: "Violence",
      overlay_nonviolence: "Non-violence",

      history_h1: "Detection History",
      history_p: "Date & time · source · label · confidence · review & download saved videos",
      history_empty: "No records yet. Upload a video or start the camera first.",
      history_count_suffix: " recent records",
      btn_delete_all: "Delete all history",
      th_id: "ID",
      th_datetime: "Date & time",
      th_source: "Source",
      th_label: "Label",
      th_confidence: "Confidence",
      th_video: "Video",
      th_actions: "Actions",
      pill_violence: "Violence",
      pill_ok: "Non-violence",
      no_video: "No video",
      btn_view: "View",
      btn_download: "Download",
      btn_delete: "Delete",
      confirm_delete_one: "Delete record #{id} and its saved video?",
      confirm_delete_all: "Delete all history and every saved video?",
      err_delete_one: "Could not delete the record.",
      err_delete_all: "Could not delete history.",
      source_upload: "Upload",
      source_webcam: "Webcam",
      source_rtsp: "RTSP",
    },
  };

  function getTheme() {
    try { return localStorage.getItem(STORAGE_THEME) || "light"; } catch (e) { return "light"; }
  }
  function getLang() {
    try { return localStorage.getItem(STORAGE_LANG) || "vi"; } catch (e) { return "vi"; }
  }
  function t(key) {
    const lang = getLang();
    const dict = I18N[lang] || I18N.vi;
    return dict[key] !== undefined ? dict[key] : (I18N.vi[key] !== undefined ? I18N.vi[key] : key);
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const btn = document.getElementById("themeToggleBtn");
    if (btn) {
      btn.textContent = theme === "dark" ? t("theme_light") : t("theme_dark");
    }
  }

  function applyLang(lang) {
    document.documentElement.lang = lang;
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      const val = (I18N[lang] || I18N.vi)[key];
      if (val !== undefined) el.textContent = val;
    });
    document.querySelectorAll("[data-i18n-title]").forEach((el) => {
      const key = el.getAttribute("data-i18n-title");
      const val = (I18N[lang] || I18N.vi)[key];
      if (val !== undefined) el.title = val;
    });
    const themeBtn = document.getElementById("themeToggleBtn");
    if (themeBtn) applyTheme(getTheme());
    const langBtn = document.getElementById("langToggleBtn");
    if (langBtn) langBtn.textContent = lang === "vi" ? "EN" : "VI";
    document.dispatchEvent(new CustomEvent("vd:langchange", { detail: { lang } }));
  }

  function init() {
    applyTheme(getTheme());
    applyLang(getLang());
    const themeBtn = document.getElementById("themeToggleBtn");
    if (themeBtn) {
      themeBtn.addEventListener("click", () => {
        const next = getTheme() === "dark" ? "light" : "dark";
        try { localStorage.setItem(STORAGE_THEME, next); } catch (e) {}
        applyTheme(next);
      });
    }
    const langBtn = document.getElementById("langToggleBtn");
    if (langBtn) {
      langBtn.addEventListener("click", () => {
        const next = getLang() === "vi" ? "en" : "vi";
        try { localStorage.setItem(STORAGE_LANG, next); } catch (e) {}
        applyLang(next);
      });
    }
  }

  window.VD = window.VD || {};
  window.VD.t = t;
  window.VD.getLang = getLang;
  window.VD.getTheme = getTheme;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
