/* =========================================
   IMAGE UPLOAD & PREVIEW
========================================= */

const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");

const imagePreviewContainer = document.getElementById(
    "imagePreviewContainer"
);

const imagePreview = document.getElementById(
    "imagePreview"
);

const removeImageBtn = document.getElementById(
    "removeImageBtn"
);

const generateCaptionBtn = document.getElementById(
    "generateCaptionBtn"
);

const captionError = document.getElementById(
    "captionError"
);

const captionErrorMessage = document.getElementById(
    "captionErrorMessage"
);


/* =========================================
   ALLOWED IMAGE TYPES
========================================= */

const allowedImageTypes = [
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp"
];


/* =========================================
   SHOW ERROR
========================================= */

function showUploadError(message) {

    captionErrorMessage.textContent = message;

    captionError.hidden = false;
}


/* =========================================
   HIDE ERROR
========================================= */

function hideUploadError() {

    captionError.hidden = true;
}


/* =========================================
   HANDLE IMAGE
========================================= */

function handleImage(file) {

    hideUploadError();

    if (!file) {
        return;
    }


    /* Validate file type */

    if (!allowedImageTypes.includes(file.type)) {

        showUploadError(
            "Please select a valid JPG, JPEG, PNG, or WEBP image."
        );

        resetImage();

        return;
    }


    /* Create image preview */

    const reader = new FileReader();

    reader.onload = function (event) {

        imagePreview.src = event.target.result;

        imagePreviewContainer.hidden = false;

        generateCaptionBtn.disabled = false;
    };

    reader.readAsDataURL(file);
}


/* =========================================
   FILE INPUT
========================================= */

imageInput.addEventListener(
    "change",
    function () {

        const file = imageInput.files[0];

        handleImage(file);
    }
);


/* =========================================
   REMOVE IMAGE
========================================= */

removeImageBtn.addEventListener(
    "click",
    function () {

        resetImage();
    }
);


/* =========================================
   RESET IMAGE
========================================= */

function resetImage() {

    imageInput.value = "";

    imagePreview.src = "";

    imagePreviewContainer.hidden = true;

    generateCaptionBtn.disabled = true;

    hideUploadError();
}


/* =========================================
   DRAG OVER
========================================= */

uploadArea.addEventListener(
    "dragover",
    function (event) {

        event.preventDefault();

        uploadArea.classList.add("drag-over");
    }
);


/* =========================================
   DRAG LEAVE
========================================= */

uploadArea.addEventListener(
    "dragleave",
    function () {

        uploadArea.classList.remove("drag-over");
    }
);


/* =========================================
   DROP IMAGE
========================================= */

uploadArea.addEventListener(
    "drop",
    function (event) {

        event.preventDefault();

        uploadArea.classList.remove("drag-over");

        const file = event.dataTransfer.files[0];

        handleImage(file);
    }
);