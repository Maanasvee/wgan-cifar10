async function generateImage(){

const response = await fetch("http://127.0.0.1:5000/generate")

const data = await response.json()

document.getElementById("result").src =
"data:image/png;base64," + data.image

}
