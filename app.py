
import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from engine import AnalysisEngine

# 設定上傳資料夾
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'supersecretkey' # 用於 flash 訊息

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_and_analyze():
    if request.method == 'POST':
        # 檢查檔案是否都存在於請求中
        if 'data_file' not in request.files or 'factor_file' not in request.files:
            flash('錯誤：請求中缺少檔案欄位')
            return redirect(request.url)
        
        data_file = request.files['data_file']
        factor_file = request.files['factor_file']

        # 檢查檔案是否被選取
        if data_file.filename == '' or factor_file.filename == '':
            flash('錯誤：請選取兩個檔案')
            return redirect(request.url)

        # 檢查並儲存檔案
        if data_file and allowed_file(data_file.filename) and factor_file and allowed_file(factor_file.filename):
            data_filename = secure_filename(data_file.filename)
            factor_filename = secure_filename(factor_file.filename)
            
            data_path = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
            factor_path = os.path.join(app.config['UPLOAD_FOLDER'], factor_filename)
            
            data_file.save(data_path)
            factor_file.save(factor_path)

            try:
                # 執行分析
                engine = AnalysisEngine(data_path=data_path, factor_path=factor_path)
                engine.run_analysis()
                summary_df = engine.get_summary_report()

                # 將結果渲染到模板
                return render_template('results.html', 
                                       table=summary_df.to_html(classes='table table-striped', index=False))
            except Exception as e:
                flash(f'分析時發生錯誤: {e}')
                return redirect(request.url)
        else:
            flash('錯誤：只允許上傳 csv 或 txt 檔案')
            return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
