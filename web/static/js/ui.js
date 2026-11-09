/* Theme (light/dark) + language (vi/en) toggle -- shared across all pages. */
(function () {
  "use strict";
  const STORAGE_THEME = "vd_theme";
  const STORAGE_LANG = "vd_lang";

  const I18N = {
    vi: {
      nav_overview: "Tổng quan",
      nav_upload: "Phân tích video",
      nav_camera: "Camera",
      nav_history: "Lịch sử cảnh báo",
      chip_api_checking: "API: Đang kiểm tra",
      chip_api_ok: "API: Kết nối",
      chip_api_down: "API: Mất kết nối",
      chip_model_checking: "Model AI: —",
      chip_model_ok: "Model AI: Kết nối",
      chip_model_err: "Model AI: Lỗi",
      system_checking: "Đang kiểm tra hệ thống",
      system_ok: "Hệ thống hoạt động",
      system_down: "Hệ thống gián đoạn",
      version_label: "Phiên bản",

      cam_name: "CAMERA 01 - Webcam",
      cam_tag: "Camera 01",
      cam_online: "Trực tuyến",
      cam_offline: "Ngoại tuyến",
      cam_placeholder: "Camera đang tắt · nhấn ▶ để bắt đầu giám sát",
      feat_title: "AI Phát Hiện Bạo Lực",
      feat_sub: "Phát hiện bạo lực – Cảnh báo kịp thời – Bảo vệ mọi người",
      feat1_t: "Nhận diện chính xác",
      feat1_d: "AI phân tích hình ảnh theo thời gian thực",
      feat2_t: "Gửi cảnh báo ngay",
      feat2_d: "Báo động và lưu video bằng chứng tự động",
      feat3_t: "Giám sát 24/7",
      feat3_d: "An toàn mọi lúc, mọi nơi",
      alert_panel_title: "CẢNH BÁO BẠO LỰC",
      alert_badge_none: "Không có",
      alert_badge_live: "Đang xử lý",
      alert_badge_saved: "Chờ xác nhận",
      alert_badge_done: "Đã xử lý",
      alert_empty: "Chưa có cảnh báo bạo lực nào.",
      alert_conf: "Độ tin cậy",
      alert_ongoing: "Đang diễn ra",
      alert_logged: "Đã lưu bằng chứng",
      alert_view: "Xem chi tiết",
      alert_ack: "Xử lý",
      stats_title: "THỐNG KÊ HỆ THỐNG",
      stats_realtime: "Cập nhật: Real-time",
      stat_alerts_today: "Cảnh báo hôm nay",
      stat_monitoring: "Đang giám sát",
      stat_uptime: "Thời gian hoạt động",
      stat_people: "Số người đang theo dõi",
      stat_risk: "Mức độ nguy cơ",
      risk_low: "Thấp",
      risk_mid: "Trung bình",
      risk_high: "Cao",
      unit_hours: "giờ",
      recent_title: "LỊCH SỬ CẢNH BÁO",
      view_all: "Xem tất cả",
      th_time: "Thời gian",
      th_type: "Loại cảnh báo",
      recent_empty: "Chưa có cảnh báo.",
      scan_idle: "Camera chưa bật",
      scan_running: "Đang quét tìm đối tượng trong khung hình…",
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
      choose_video_title: "Kéo thả video vào đây",
      choose_video_hint: "hoặc nhấn để chọn tệp từ máy tính",
      btn_detect: "Phát hiện bạo lực",
      btn_reset: "Làm mới",
      btn_new_video: "Nhập video mới",
      btn_export_report: "Xuất báo cáo",
      mail_sent: "Đã gửi email cảnh báo",
      mail_failed: "Chưa gửi được email cảnh báo — kiểm tra cấu hình SMTP",
      alert_violence: "⚠ CẢNH BÁO BẠO LỰC ĐÃ PHÁT HIỆN",
      no_result_yet: "Chưa có kết quả",
      ready_to_detect: "Sẵn sàng nhận diện",
      detecting: "Đang nhận diện và gán nhãn video...",
      annotated_note: "Video kết quả: khung đỏ là đoạn bạo lực, khung xanh là không bạo lực.",
      choose_video_first: "Chọn video trước",
      result_label: "Kết quả: ",
      confidence_label: "Độ tin cậy: ",
      saved_history_prefix: "Đã lưu lịch sử #",
      attention_label: "Attention (frame quyết định)",
      timeline_label: "Xác suất bạo lực theo thời gian (vạch ngang: ngưỡng)",
      no_segments: "Không có đoạn nào vượt ngưỡng đủ lâu.",
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
      nav_stats: "Thống kê",
      nav_settings: "Cài đặt",
      stat_alerts_total: "Tổng số cảnh báo",
      th_camera: "Camera",
      th_status: "Trạng thái",
      source_upload_cam: "Video tải lên",
      stats_page_h1: "Thống kê",
      stats_page_p: "Tổng hợp từ lịch sử nhận diện đã lưu.",
      stat_total_checks: "Tổng lượt nhận diện",
      stat_avg_conf: "Độ tin cậy TB (cảnh báo)",
      stats_7days: "CẢNH BÁO 7 NGÀY QUA",
      stats_by_source: "CẢNH BÁO THEO NGUỒN",
      th_date: "Ngày",
      th_alerts: "Cảnh báo",
      th_checks: "Lượt nhận diện",
      checks_suffix: "lượt",
      settings_h1: "Cài đặt",
      settings_p: "Giao diện, ngôn ngữ và thông tin mô hình đang chạy.",
      settings_display: "HIỂN THỊ",
      settings_theme: "Giao diện sáng / tối",
      settings_theme_d: "Lưu trên trình duyệt này",
      settings_lang: "Ngôn ngữ",
      settings_lang_d: "Tiếng Việt / English",
      settings_model: "MÔ HÌNH AI",
      settings_cam_model: "Model camera",
      settings_upload_model: "Model phân tích video",
      settings_threshold: "Ngưỡng cảnh báo camera",
      settings_movinet: "MoViNet chống báo nhầm",
      settings_device: "Thiết bị",
      on: "Bật",
      off: "Tắt",
      up_panel_title: "PHÂN TÍCH VIDEO",
      up_state_empty: "Chưa chọn video",
      up_state_ready: "Sẵn sàng",
      up_state_busy: "Đang phân tích",
      choose_video_formats: "MP4 · AVI · MOV · MKV · WEBM",
      no_file: "Chưa có tệp",
      up_result_title: "KẾT QUẢ PHÂN TÍCH",
      up_waiting: "Đang chờ video",
      up_segments: "Đoạn bạo lực",
      up_violent_time: "Thời gian bạo lực",
      up_history_id: "Mã lịch sử",
      up_info_title: "THÔNG TIN VIDEO",
      up_name: "Tên tệp",
      up_size: "Dung lượng",
      up_duration: "Thời lượng",
      up_resolution: "Độ phân giải",
      up_steps_title: "CÁCH SỬ DỤNG",
      up_step1_t: "Chọn video",
      up_step1_d: "Kéo thả hoặc nhấn vào khung video",
      up_step2_t: "Phát hiện bạo lực",
      up_step2_d: "Model chấm từng cửa sổ 16 khung hình",
      up_step3_t: "Xem lại kết quả",
      up_step3_d: "Nhấn đoạn bên dưới biểu đồ để tua tới đó",
      up_verdict_bad: "Phát hiện hành vi bạo lực",
      up_verdict_ok: "Không phát hiện bạo lực",
    },
    en: {
      nav_overview: "Overview",
      nav_upload: "Video analysis",
      nav_camera: "Camera",
      nav_history: "Alert history",
      chip_api_checking: "API: Checking",
      chip_api_ok: "API: Connected",
      chip_api_down: "API: Disconnected",
      chip_model_checking: "AI model: —",
      chip_model_ok: "AI model: Connected",
      chip_model_err: "AI model: Error",
      system_checking: "Checking system",
      system_ok: "System online",
      system_down: "System interrupted",
      version_label: "Version",

      cam_name: "CAMERA 01 - Webcam",
      cam_tag: "Camera 01",
      cam_online: "Online",
      cam_offline: "Offline",
      cam_placeholder: "Camera is off · press ▶ to start monitoring",
      feat_title: "AI Violence Detection",
      feat_sub: "Detect violence – Alert in time – Keep people safe",
      feat1_t: "Accurate recognition",
      feat1_d: "AI analyses the picture in real time",
      feat2_t: "Instant alerts",
      feat2_d: "Raises the alarm and saves video evidence",
      feat3_t: "24/7 monitoring",
      feat3_d: "Safe anytime, anywhere",
      alert_panel_title: "VIOLENCE ALERT",
      alert_badge_none: "None",
      alert_badge_live: "Open",
      alert_badge_saved: "Awaiting review",
      alert_badge_done: "Handled",
      alert_empty: "No violence alerts yet.",
      alert_conf: "Confidence",
      alert_ongoing: "Ongoing",
      alert_logged: "Evidence saved",
      alert_view: "View details",
      alert_ack: "Mark handled",
      stats_title: "SYSTEM STATISTICS",
      stats_realtime: "Updated: Real-time",
      stat_alerts_today: "Alerts today",
      stat_monitoring: "Monitoring",
      stat_uptime: "Uptime",
      stat_people: "People tracked",
      stat_risk: "Risk level",
      risk_low: "Low",
      risk_mid: "Medium",
      risk_high: "High",
      unit_hours: "h",
      recent_title: "ALERT HISTORY",
      view_all: "View all",
      th_time: "Time",
      th_type: "Alert type",
      recent_empty: "No alerts yet.",
      scan_idle: "Camera is off",
      scan_running: "Scanning the scene for people…",
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
      choose_video_title: "Drop a video here",
      choose_video_hint: "or click to choose a file from your computer",
      btn_detect: "Detect Violence",
      btn_reset: "Reset",
      btn_new_video: "Import new video",
      btn_export_report: "Export report",
      mail_sent: "Alert email sent",
      mail_failed: "Alert email could not be sent — check SMTP settings",
      alert_violence: "⚠ VIOLENCE ALERT DETECTED",
      no_result_yet: "No result yet",
      ready_to_detect: "Ready to detect",
      detecting: "Detecting and labelling the video...",
      annotated_note: "Result video: red boxes mark violent segments, green boxes non-violent ones.",
      choose_video_first: "Please choose a video first",
      result_label: "Result: ",
      confidence_label: "Confidence: ",
      saved_history_prefix: "Saved to history #",
      attention_label: "Attention (deciding frame)",
      timeline_label: "Violence probability over time (line: threshold)",
      no_segments: "No segment stayed above the threshold long enough.",
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
      nav_stats: "Statistics",
      nav_settings: "Settings",
      stat_alerts_total: "Total alerts",
      th_camera: "Camera",
      th_status: "Status",
      source_upload_cam: "Uploaded video",
      stats_page_h1: "Statistics",
      stats_page_p: "Summary of the saved detection history.",
      stat_total_checks: "Total detections",
      stat_avg_conf: "Avg. confidence (alerts)",
      stats_7days: "ALERTS, LAST 7 DAYS",
      stats_by_source: "ALERTS BY SOURCE",
      th_date: "Date",
      th_alerts: "Alerts",
      th_checks: "Detections",
      checks_suffix: "runs",
      settings_h1: "Settings",
      settings_p: "Appearance, language and the running model.",
      settings_display: "DISPLAY",
      settings_theme: "Light / dark theme",
      settings_theme_d: "Saved in this browser",
      settings_lang: "Language",
      settings_lang_d: "Tiếng Việt / English",
      settings_model: "AI MODEL",
      settings_cam_model: "Camera model",
      settings_upload_model: "Video analysis model",
      settings_threshold: "Camera alert threshold",
      settings_movinet: "MoViNet false-alarm veto",
      settings_device: "Device",
      on: "On",
      off: "Off",
      up_panel_title: "VIDEO ANALYSIS",
      up_state_empty: "No video",
      up_state_ready: "Ready",
      up_state_busy: "Analysing",
      choose_video_formats: "MP4 · AVI · MOV · MKV · WEBM",
      no_file: "No file",
      up_result_title: "ANALYSIS RESULT",
      up_waiting: "Waiting for a video",
      up_segments: "Violent segments",
      up_violent_time: "Violent time",
      up_history_id: "History ID",
      up_info_title: "VIDEO INFO",
      up_name: "File name",
      up_size: "Size",
      up_duration: "Duration",
      up_resolution: "Resolution",
      up_steps_title: "HOW TO USE",
      up_step1_t: "Choose a video",
      up_step1_d: "Drop it or click the video area",
      up_step2_t: "Detect violence",
      up_step2_d: "The model scores every 16-frame window",
      up_step3_t: "Review the result",
      up_step3_d: "Click a segment under the chart to jump there",
      up_verdict_bad: "Violent behaviour detected",
      up_verdict_ok: "No violence detected",
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

  function setKey(el, key) {
    if (!el) return;
    el.setAttribute("data-i18n", key);
    el.textContent = t(key);
  }

  function tickClock() {
    const el = document.getElementById("clockChip");
    if (!el) return;
    const d = new Date();
    const p = (n) => String(n).padStart(2, "0");
    el.textContent = `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
  }

  // Header chips and sidebar status from /health: API reachable, model loaded.
  async function checkHealth() {
    let apiUp = false;
    let modelOk = false;
    try {
      const r = await fetch("/health", { cache: "no-store" });
      apiUp = r.ok;
      if (r.ok) modelOk = Boolean((await r.json()).ok);
    } catch (e) { apiUp = false; }
    const setDot = (chipId, cls) => {
      const dot = document.querySelector(`#${chipId} .dot`);
      if (dot) dot.className = `dot ${cls}`;
    };
    setDot("apiChip", apiUp ? "ok" : "bad");
    setKey(document.getElementById("apiChipText"), apiUp ? "chip_api_ok" : "chip_api_down");
    setDot("modelChip", modelOk ? "ok" : apiUp ? "bad" : "");
    setKey(document.getElementById("modelChipText"), modelOk ? "chip_model_ok" : apiUp ? "chip_model_err" : "chip_model_checking");
    const sideDot = document.getElementById("sideDot");
    if (sideDot) sideDot.className = `dot ${apiUp && modelOk ? "ok" : "bad"}`;
    setKey(document.getElementById("sideStatusText"), apiUp && modelOk ? "system_ok" : "system_down");
  }

  function init() {
    applyTheme(getTheme());
    applyLang(getLang());
    tickClock();
    setInterval(tickClock, 1000);
    if (document.getElementById("apiChip")) {
      checkHealth();
      setInterval(checkHealth, 15000);
    }
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
