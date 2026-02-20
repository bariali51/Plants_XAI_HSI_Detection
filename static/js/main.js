const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const analyzeBtn = document.getElementById('analyzeBtn');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const uploadSection = document.querySelector('.upload-section');

let selectedFile = null;

// 1. التعامل مع اختيار الملف
fileInput.addEventListener('change', handleFileSelect);

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        showFileInfo(file.name);
        analyzeBtn.disabled = false;
    }
}

// 2. التعامل مع السحب والإفلات (Drag & Drop)
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) {
        selectedFile = file;
        fileInput.files = e.dataTransfer.files;
        showFileInfo(file.name);
        analyzeBtn.disabled = false;
    }
});

function showFileInfo(name) {
    fileName.textContent = `📄 ${name}`;
    fileInfo.style.display = 'flex';
}

function removeFile() {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    analyzeBtn.disabled = true;
}

// 3. إرسال الطلب للسيرفر
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // إخفاء قسم الرفع وإظهار التقدم
    uploadSection.style.display = 'none';
    progressSection.style.display = 'block';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        // إرسال الملف إلى Django View
        const response = await fetch('/analyze/', {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': getCookie('csrftoken') // دالة مساعدة للأمان
            }
        });

        if (!response.ok) throw new Error('فشل التحليل');

        const data = await response.json();

        // محاكاة شريط التقدم (لأن المعالجة قد تكون سريعة)
        simulateProgress(() => {
            showResults(data);
        });

    } catch (error) {
        alert('حدث خطأ: ' + error.message);
        location.reload();
    }
});

// 4. عرض النتائج
function showResults(data) {
    progressSection.style.display = 'none';
    resultsSection.style.display = 'block';

    // عرض التنبؤ
    document.getElementById('predictionResult').textContent = data.prediction || 'غير محدد';

    // عرض شريط الثقة
    const confidence = (data.confidence * 100).toFixed(1);
    document.getElementById('confidenceFill').style.width = `${confidence}%`;
    document.getElementById('confidenceText').textContent = `${confidence}%`;

    // عرض الخريطة الحرارية (إذا أرسلها الباك إند)
    if (data.heatmap_url) {
        document.getElementById('heatmapImage').src = data.heatmap_url;
    }

    // رسم مخطط XAI باستخدام Plotly
    if (data.xai_labels && data.xai_values) {
        const trace = {
            x: data.xai_values,
            y: data.xai_labels,
            type: 'bar',
            orientation: 'h',
            marker: { color: '#2ecc71' }
        };
        const layout = {
            title: 'أهمية الأطوال الموجية (Feature Importance)',
            xaxis: { title: 'قيمة التأثير (SHAP Value)' },
            margin: { l: 100, r: 20, t: 50, b: 50 }
        };
        Plotly.newPlot('xaiChart', [trace], layout);
    }
}

// 5. محاكاة شريط التقدم
function simulateProgress(callback) {
    let width = 0;
    const bar = document.getElementById('progressFill');
    const interval = setInterval(() => {
        if (width >= 100) {
            clearInterval(interval);
            callback();
        } else {
            width += 5;
            bar.style.width = width + '%';
        }
    }, 100);
}

// دالة مساعدة لجلب CSRF Token من الكوكيز
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}