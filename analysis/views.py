# classifier/views.py
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import uuid
from .ml_utils import classifier_instance


def upload_image(request):
    context = {
        'result': None,
        'image_url': None,
        'gradcam_url': None,
        'error': None
    }

    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        fs = FileSystemStorage()

        try:
            # 1. التنبؤ الأساسي
            result = classifier_instance.predict(image_file)
            context['result'] = result

            # 2. حفظ الصورة الأصلية
            unique_id = uuid.uuid4().hex[:8]
            original_filename = f"{unique_id}_original_{image_file.name}"
            original_path = fs.save(original_filename, image_file)
            context['image_url'] = fs.url(original_path)

            # 3. توليد Grad-CAM
            image_file.seek(0)
            gradcam_result = classifier_instance.apply_gradcam(image_file)

            if gradcam_result:
                # ✅ حفظ صورة Grad-CAM على الخادم (بدلاً من base64)
                gradcam_filename = f"{unique_id}_gradcam.png"
                gradcam_path = os.path.join(settings.MEDIA_ROOT, gradcam_filename)
                gradcam_result['superimposed'].save(gradcam_path, format="PNG")
                context['gradcam_url'] = settings.MEDIA_URL + gradcam_filename
                context['gradcam_class'] = gradcam_result['predicted_class'].replace('_', ' ')

                print(f"✅ Grad-CAM saved: {gradcam_filename}")

            print(f"✅ Prediction: {result['disease']} ({result['confidence']})")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            context['error'] = str(e)

    return render(request, 'analysis/upload.html', context)