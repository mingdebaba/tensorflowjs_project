console.log("where am i")

/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */

async function getData() {
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataResponse.json();
    const cleaned = carsData.map(car => ({
        mpg:car.Miles_per_Gallon,
        horsepower:car.Horsepower,
    }))
    .filter(car => (car.mpg != null && car.horsepower != null));

    return cleaned;
    console.log(cleaned);
}
getData();


async function run(){
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    const values = data.map(d => ({
        x: d.horsepower,
        y:d.mpg,
    }));
    tfvis.render.scatterplot(
        {name: 'Horsepower v MPG'},
        {values},
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    )
    // Create the model
    const model = createModel();
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    
    const tensorData = convertToTensor(data);
    const {inputs,labels} = tensorData;
    // Train the model
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    // Make some predictions using the model and compare them to the
    // original data
    testModel(model, data, tensorData);

}

document.addEventListener('DOMContentLoaded',run)


function createModel(){
    const model = tf.sequential();
     // Add a single input layer
    //  /inputShape 是 [1]，因为我们将 1 数字用作输入（某辆指定汽车的马力）。
     //units 用于设置权重矩阵在层中的大小。将其设置为 1 即表示数据的每个输入特征的权重为 1。
     //密集层带有一个偏差项，因此我们无需将 useBias 设置为 true，并在对 tf.layers.dense 的后续调用中省略这一步
    model.add(tf.layers.dense({inputShape:[1],units:1,useBias:true}));
     // Add an output layer
     //
     model.add(tf.layers.dense({units:1,useBias:true}));

     return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data){
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.
    return tf.tidy(() => {
        // Step 1. Shuffle the data
        //数据重排
        //不学习纯粹依赖于数据输入顺序的东西
        //对子组中的结构不敏感（例如，如果模型在训练的前半部分仅看到高马力汽车，
        //可能会学习一种不适用于数据集其余部分的关系）。
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor
        const inputs = data.map(d => d.horsepower)
        const labels = data.map(d => d.mpg);
        //转换为张量
        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels,[labels.length,1]);

        //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        //我们会对数据进行归一化。我们使用最小-最大缩放比例将数据归一化为数值范围 0-1。归一化至关重要，
        //因为您将使用 tensorflow.js 构建的许多机器学习模型的内部构件旨在处理不太大的数字
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        //返回数据和归一化边界
        return {
            inputs:normalizedInputs,
            labels:normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    });
}

    //完成将数据表示为张量以后，就可以开始训练过程了
async function trainModel(model,inputs,labels){
    // Prepare the model for training.
    // /训练模型之前，我们必须对其进行“编译"
    //optimizer 控制模型更新的算法
    //loss 告知模型在学习各个批次表现如何
    //meanSquaredError 将模型与预测的真实值作比较
    model.compile({
        optimizer:tf.train.adam(),
        loss:tf.losses.meanSquaredError,
        metrics:['mse'],
    });
    //batchSize 是指模型在每次训练迭代时会看到的数据子集的大小。
    //epochs 表示模型查看您提供的整个数据集的次数
    const batchSize = 32;
    const epochs = 50;
    //启动模型
    //model.fit 是您为了启动训练循环而调用的函数。
    //这是一个异步函数，因此我们会返回它提供的 promise，以便调用方确定训练何时完成。
    return await model.fit(inputs,labels,{
        batchSize,
        epochs,
        shuffle:true,
        callbacks:tfvis.show.fitCallbacks(
            {name:'training performance'},
            ['loss','mse'],
            {height:200,callbacks:['onEpochEnd']}
        )
    })
}


function testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;
  
    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
  
      const xs = tf.linspace(0, 1, 100);
      const preds = model.predict(xs.reshape([100, 1]));
  
      const unNormXs = xs
        .mul(inputMax.sub(inputMin))
        .add(inputMin);
  
      const unNormPreds = preds
        .mul(labelMax.sub(labelMin))
        .add(labelMin);
  
      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });
  
    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: preds[i]}
    });
  
    const originalPoints = inputData.map(d => ({
      x: d.horsepower, y: d.mpg,
    }));
  
    tfvis.render.scatterplot(
      {name: 'Model Predictions vs Original Data'},
      {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
      {
        xLabel: 'Horsepower',
        yLabel: 'MPG',
        height: 300
      }
    );
  }