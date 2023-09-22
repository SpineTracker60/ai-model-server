from flask import Blueprint

# 추천 시스템 bp
bp = Blueprint('rs',__name__,url_prefix='/rs')

# 
@bp.route('/updateDB')
def re_learning():
    return 'DB 업데이트/재학습 페이지입니다!'