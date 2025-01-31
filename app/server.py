from flask import Flask, request, render_template, url_for
import os, sys

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def add_project_paths():
    """
        커스텀 패키지를 인식하도록 sys.path에 패키지에 경로를 추가함.
    """
    cwd = os.getcwd()
    main_dir = os.path.join(cwd)
    print(main_dir)

    paths = [main_dir]
    for path in paths:
        sys.path.append(path)


add_project_paths()

from search_image import search_image


app = Flask(__name__, static_folder=os.path.join(SCRIPT_PATH, "../data/flickr30k/Images"))

@app.route('/', methods=["GET", "POST"])
def searh_home():
    result_image="No image"
    if request.method == 'POST':
        search_query = request.form.get('query')
        print(search_query)
        result_image = search_image(search_query)
        print(result_image)

    return render_template("index.html", result_image=result_image)


if __name__=="__main__":
    app.run(debug=True)