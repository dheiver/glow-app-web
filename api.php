<?php

// Requisitos:
// - Laravel 8.x
// - Intervention Image 2.x
// - Keras for PHP 3.x
// - TensorFlow 2.x com suporte a Keras (GPU recomendado)

use Illuminate\Http\Request;

// Passo a passo para executar:
// 1. Certifique-se de que os requisitos acima estejam instalados e configurados corretamente.
// 2. Crie um novo projeto Laravel ou abra um projeto existente.
// 3. Coloque este arquivo em um diretório adequado dentro do projeto (por exemplo, app/Http/Controllers/).
// 4. Abra o arquivo routes/web.php e adicione a seguinte rota:
//    Route::post('/classify', 'App\Http\Controllers\ClassifyController@classify');
// 5. Crie um novo controlador usando o comando do Artisan:
//    php artisan make:controller ClassifyController
// 6. Abra o arquivo app/Http/Controllers/ClassifyController.php e cole o código acima no método classify().
// 7. Certifique-se de que o modelo de classificação (BreastCancerSegmentor.h5) esteja no diretório de modelos Keras do seu projeto.
// 8. Inicie o servidor Laravel usando o comando do Artisan:
//    php artisan serve
// 9. Envie uma imagem para a rota /classify usando uma ferramenta como o Postman ou através de um formulário HTML.

class ClassifyController extends Controller
{
    public function classify(Request $request)
    {
        if (!$request->hasFile('file')) {
            return response()->json(['error' => 'No file uploaded']);
        }

        $image = $request->file('file');
        $imagePath = $image->store('images', 'public');
        $imageFullPath = storage_path('app/public/' . $imagePath);

        $originalImage = Image::make($imageFullPath);

        // Melhoria 1: verifica se a imagem está em modo RGB
        if (!$originalImage->isTrueColor()) {
            $originalImage = $originalImage->convert('RGB');
        }

        // Melhoria 2: aplica um filtro de nitidez na imagem antes do redimensionamento
        $originalImage = $originalImage->filter(function ($pixel) {
            return $pixel->sharpen(50);
        });

        $resizedImage = $originalImage->resize(224, 224);
        $finalImage = $resizedImage->normalize();

        $imageData = array_values($finalImage->toArray());
        $inputData = [$imageData];

        $model = app('keras')->model->load('BreastCancerSegmentor.h5');
        $prediction = $model->predict($inputData);
        $classIdx = array_keys($prediction, max($prediction))[0];
        $classLabel = ($classIdx == 0) ? 'benign' : (($classIdx == 1) ? 'normal' : 'malignant');

        if (max($prediction) < 0.9) {
            return response()->json(['error' => 'Unable to classify image with high confidence.']);
        }

        return response()->json(['class' => $classLabel, 'probability' => floatval(max($prediction))]);
    }
}
