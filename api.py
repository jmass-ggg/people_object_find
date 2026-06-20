from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from search_service import BlockedQueryError, ImageSearchService


BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "result"
TEMPLATE_DIR = BASE_DIR / "templates"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
TEMPLATE_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024


app = FastAPI(
    title="Lost and Found Visual Search",
    description="Upload an object image and get the top 10 visually similar dataset matches.",
)

app.mount(
    "/result",
    StaticFiles(directory=str(RESULT_DIR)),
    name="result",
)

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


search_service = ImageSearchService(
    index_file=str(BASE_DIR / "image_index.faiss"),
    metadata_file=str(BASE_DIR / "image_metadata.pkl"),
    result_root=str(RESULT_DIR / "api"),
    top_k=10,
    rerank_limit=100,
    allow_different_class=False,
)


def render_page(
    request: Request,
    query=None,
    results=None,
    error=None,
    status_code=200,
):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "query": query,
            "results": results,
            "error": error,
        },
        status_code=status_code,
    )


async def save_uploaded_image(file: UploadFile) -> Path:
    original_filename = file.filename or ""
    extension = Path(original_filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, PNG, and WEBP images are allowed.",
        )

    upload_path = UPLOAD_DIR / f"{uuid4().hex}{extension}"

    total_size = 0

    with upload_path.open("wb") as output_file:
        while True:
            chunk = await file.read(1024 * 1024)

            if not chunk:
                break

            total_size += len(chunk)

            if total_size > MAX_UPLOAD_SIZE:
                upload_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail="Image is too large. Maximum upload size is 10 MB.",
                )

            output_file.write(chunk)

    return upload_path


@app.get("/")
async def home(request: Request):
    return render_page(request)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_size": search_service.index.ntotal,
    }


@app.post("/search")
async def search_web(request: Request, file: UploadFile = File(...)):
    try:
        upload_path = await save_uploaded_image(file)

        data = search_service.search(
            query_image_path=upload_path,
            top_k=10,
        )

        return render_page(
            request=request,
            query=data["query"],
            results=data["results"],
        )

    except BlockedQueryError as e:
        return render_page(
            request=request,
            error=str(e),
            status_code=200,
        )


@app.post("/api/search")
async def search_api(file: UploadFile = File(...)):
    try:
        upload_path = await save_uploaded_image(file)

        data = search_service.search(
            query_image_path=upload_path,
            top_k=10,
        )

        return JSONResponse(data)

    except BlockedQueryError as e:
        return JSONResponse(
            {
                "blocked": True,
                "reason": str(e),
                "query": None,
                "results": [],
            },
            status_code=200,
        )
