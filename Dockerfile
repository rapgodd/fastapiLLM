# 1. 사용할 베이스 이미지 (Python 3.9 버전)
FROM python:3.9

# 2. 컨테이너 안에서 작업할 디렉터리를 /app으로 설정
WORKDIR /app

# 3. requirements.txt를 먼저 복사해서 라이브러리 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. 모든 소스 코드 복사(여기서는 main.py, .env, 기타 py파일, docx파일 등)
COPY . .

# 5. (선택) FastAPI를 8000포트로 열 것임 -> Docker 내부 포트
EXPOSE 8000

# 6. 컨테이너 실행 시 uvicorn으로 서버 실행
#    "main:app" -> main.py 안에 정의된 "app" 객체
#    --host 0.0.0.0 -> 컨테이너 내부 IP (외부에서 접근 가능)
#    --port 8000 -> 포트
CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]
