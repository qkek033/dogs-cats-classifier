# 무료 배포 가이드 (Render + Streamlit Cloud)

아래 순서대로 하면 된다. 계정만 있으면 전부 무료.

---

## 1. 모델을 GitHub Release에 올리기

**방법 A: 스크립트 사용 (GitHub CLI 필요)**

1. [GitHub CLI](https://cli.github.com/) 설치 후 터미널에서 `gh auth login` 로 로그인.
2. 프로젝트 루트에서 실행:
   ```bash
   scripts\upload_model_to_github_release.bat
   ```
3. 완료되면 GitHub 레포 → **Releases** → **v1.0.0** 들어가서 `best_model.pth` 항목 **우클릭 → 링크 주소 복사**.  
   (예: `https://github.com/사용자명/레포명/releases/download/v1.0.0/best_model.pth`)  
   이 URL을 **메모장에 붙여넣어 두기** — 2단계에서 쓴다.

**방법 B: 수동으로 올리기**

1. GitHub 레포 페이지 → **Releases** → **Create a new release**.
2. Tag: `v1.0.0`, Title: `Model v1.0.0` 등으로 입력.
3. **Attach binaries**에서 로컬의 `models/checkpoints/best_model.pth` 선택 후 업로드.
4. **Publish release** 클릭.
5. 올라간 `best_model.pth` **우클릭 → 링크 주소 복사**해서 메모장에 저장.

---

## 2. Render에 API 배포

1. [render.com](https://render.com) 가입/로그인.
2. **Dashboard** → **New +** → **Web Service**.
3. **Connect a repository**에서 이 프로젝트 GitHub 레포 연결 (권한 허용).
4. 설정:
   - **Name**: `dogcat-api` (원하는 이름)
   - **Region**: Singapore 또는 가까운 곳
   - **Branch**: `main`
   - **Runtime**: **Docker**
   - **Instance type**: **Free**

5. **Environment** 섹션에서 **Add Environment Variable**:
   - Key: `MODEL_URL`
   - Value: 1단계에서 복사한 URL (예: `https://github.com/.../releases/download/v1.0.0/best_model.pth`)

6. **Create Web Service** 클릭.
7. 빌드가 끝날 때까지 대기 (몇 분 걸릴 수 있음). 끝나면 상단에 **URL**이 생김 (예: `https://dogcat-api-xxxx.onrender.com`).  
   이 URL을 **복사해서 메모장에 저장** — 3단계에서 쓴다.

8. (선택) 무료 플랜은 15분 미사용 시 sleep이라, 첫 요청이 느릴 수 있음. 테스트할 때 한 번 `/health` 호출해 보면 된다.

---

## 3. Streamlit Community Cloud에 UI 배포

1. [share.streamlit.io](https://share.streamlit.io) 가입/로그인 (GitHub 연동).
2. **New app** 클릭.
3. 설정:
   - **Repository**: 이 프로젝트 레포 선택
   - **Branch**: `main`
   - **Main file path**: `ui/streamlit_app.py`

4. **Advanced settings** 열기:
   - **Secrets** 또는 환경변수에 다음 추가 (Streamlit은 TOML 형식으로 넣는 경우가 많음. 화면에 맞게):
     - `API_URL` = 2단계에서 복사한 Render URL (예: `https://dogcat-api-xxxx.onrender.com`)
   - Requirements file을 지정할 수 있으면 `requirements-ui.txt` 로 두면 빌드가 가벼움. 없으면 루트 `requirements.txt` 사용.

5. **Deploy** 클릭.
6. 빌드가 끝나면 앱 URL이 나옴 (예: `https://xxx.streamlit.app`). 이 주소가 데모 링크다.

---

## 4. 확인

- Streamlit 앱 주소 접속 → 이미지 업로드 → Classify → 결과 나오면 성공.
- API가 sleep 상태면 첫 요청에 30초~1분 걸릴 수 있음. 그 다음부터는 정상 속도.

---

## 요약 체크리스트

- [ ] 1. `best_model.pth` 를 GitHub Release에 올리고 다운로드 URL 복사
- [ ] 2. Render에서 Web Service 생성, Docker, 환경변수 `MODEL_URL` 설정
- [ ] 3. Render 배포 완료 후 API URL 복사
- [ ] 4. Streamlit Cloud에서 New app, Main file `ui/streamlit_app.py`, 환경변수 `API_URL` 설정
- [ ] 5. Streamlit 앱에서 이미지 분류 테스트
