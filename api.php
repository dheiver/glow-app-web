<?php

use Illuminate\Http\Request;

Route::post('/classify', function (Request $request) {
    if (!$request->hasFile('file')) {
        return response()->json(['error' => 'No file uploaded']);
    }

    $image = $request->file('file');
    $imagePath = $image->store('images', 'public');
    $imageFullPath = storage_path('app/public/' . $imagePath);

    $originalImage = Image::make($imageFullPath);
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
});
