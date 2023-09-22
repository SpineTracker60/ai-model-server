from flask import Blueprint

# 게시판 bp
bp = Blueprint('board',__name__,url_prefix='/board')

# 게시판 작성 글 분류
# 1) 악의적 제품 리뷰
# 2) 도박/유흥 광고
# 3) etc...

@bp.route('/writingClassification')
def writing_Classify():
    return '게시판 작성 글 분류페이지 입니다!'