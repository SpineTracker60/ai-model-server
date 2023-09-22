from flask import Blueprint

# 기본 bp
bp = Blueprint('main',__name__,url_prefix='/')

@bp.route('/')
def hello_main():
    return '기본 페이지입니다!!!!'