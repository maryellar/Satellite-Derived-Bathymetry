// Section 1: loading in the data
// Define your study area
var golovinGeom = ee.Geometry.Polygon(
    [[[-163.46897712147512, 64.71290872669046],
      [-163.46897712147512, 64.43223975287069],
      [-162.79469123280327, 64.43223975287069],
      [-162.79469123280327, 64.71290872669046]]], null, false);

// Load in your DEM (topobathy data), filtering by the region of interest (study area)
var golovinBay = ee.ImageCollection("projects/uaf-coastal-mapping/assets/USACE_AK_DEM_Mosaic")
.filterBounds(golovinGeom);

// Set visualization parameters for the DEM data
var golVisParams = {
    bands: ['b1'], 
    min: -3.5, 
    max: 0, 
    gamma: .8 
};

// Add DEM data as a map layer
Map.addLayer({
    eeObject: golovinBay, 
    visParams: golVisParams, 
    name: 'Golovin Bay initial',
    shown: true
});


// Load satellite imagery (here we use the Harmonized Landsat and Sentinel-2 satellite data)
// Below we have filtered by date, geometry, and cloud coverage in order to find an image that is the most clear
var HLS = ee.ImageCollection("NASA/HLS/HLSL30/v002")
    .filterDate('2019-06-01', '2019-09-30') //filter data from June to September for months with maximum daylight and minimum snowcover
    .filterBounds(golovinGeom)
    .filter(ee.Filter.lt('CLOUD_COVERAGE', 30));

// Set your visualization parameters and the layer to the map to see how the image looks 
var HLSvisParams = {
    bands: ['B4', 'B3', 'B2'],
    min:0.01,
    max:0.1,
};
Map.addLayer(HLS, HLSvisParams, 'HLS RGB bands', false);

// Section 2: Preprocessing Functions

// Cloud mask function:
// The HLS product uses the FMask algorithm 
function maskHSL(image) {
    // Bit 0 - Fill
    // Bit 1 - Dilated Cloud
    // Bit 2 - Cirrus
    // Bit 3 - Cloud
    // Bit 4 - Cloud Shadow
    var qaMask = image.select('Fmask').bitwiseAnd(parseInt('11111', 2)).eq(0);
    // Apply masks
    return image.updateMask(qaMask)
}

// Apply cloud mask to each image in the collection and add to map
var masked = HLS.map(maskHSL)
Map.addLayer(masked, HLSvisParams, 'Masked HLS', true);

// Find the least cloudy image in the HLS image collection
// Look through the objects in the HLS collection and find the object with the lowest cloud coverage score that also fills your region of interest
print(HLS, 'HLS: image collection');
var imageList = HLS.toList(HLS.size());
var firstImage = ee.Image(imageList.get(0));

// Apply cloud mask to the least cloudy image in the collection that you found above
var maskedImg = maskHSL(firstImage);
Map.addLayer(maskedImg, HLSvisParams, 'HLS: masked first image', false)


// Manually draw your own geometries used for training the water extraction model
// Turn both geometries into feature collection and give them a class property
var landFeature = ee.Feature(land, {'class': 0});
var waterFeature = ee.Feature(water, {'class': 1});

// Create land and water feature collections then merge into a single collection
var landCollection = ee.FeatureCollection([landFeature]);
var waterCollection = ee.FeatureCollection([waterFeature]);
var trainingCollection = landCollection.merge(waterCollection);

// Land mask function: uses the land and water geometries to mask land and classify water using the Modified Normalized Difference Water Index (MNDWI)
function landmask(img) {
    // MNDWI uses SWIR1(B6) and green(B3) spectral bands, these bands could be different depending on the satellite imagery you use 
    // MNDWI = (Green - SWIR) / (Green + SWIR)
    var mndwi = img.normalizedDifference(['B3', 'B6']); 
    var training = mndwi.sampleRegions(trainingCollection, ['class'], 3); 
    var trained = ee.Classifier.smileRandomForest(10)
        .train(training, 'class');  //smileRandomForest creates an empty Random Forest classifier with ten decision trees.
    var classified = mndwi.classify(trained); 
    var mask = classified.eq(1); 

    return img.updateMask(mask);
}

// Apply the land masking function to the image you chose above from the HLS data
maskedImg = landmask(maskedImg); 
Map.addLayer(maskedImg, HLSvisParams, 'land masked image', true); // adds masked image to the map



// DIV procedure: creates a proxy image that helps minimize the bias of the spectral values due to the water column during classification and
// bathymetry procedures
function kernel(img) {
    var boxcar = ee.Kernel.square({ // generates a square-shaped boolean kernel.
        radius: 2, 
        units: 'pixels',
        normalize: true
    });
    return img.convolve(boxcar); // convolves each band of an image with the given kernel.
}

function makePositive(img) { // performs conditional replacement of values to ensure that all values are positive
return img.where(img.lte(0), 0.0001);
}

// div function: performs a series of operations to calculate depth-invariant indices for different band combinations.
// Involves logarithmic transformations, covariance calculation, and mathematical operations on image bands.
function div(img) { 
    var band1 = ee.List(['B2', 'B3', 'B4', 'B2', 'B3']); 
    var band2 = ee.List(['B4', 'B4', 'B3', 'B3', 'B2']); 
    var nband = ee.List(['B2B4', 'B3B4', 'B4B3', 'B2B3', 'B3B2']); 

    // Iterate over the above band combinations performing logarithmic transformations and  calculating the covariance, attenuation coefficient ratio, and 
    // the depth-invariant index for the band pair
    for (var i = 0; i < 5; i += 1) { 
        var x = band1.get(i); 
        var y = band2.get(i);
        var z = nband.get(i);
        
        // Logarithmic transformation
        var imageLog = img.select([x, y]).log(); 

        // Calculate the covariance of the logarithmically transformed bands
        var covariance = imageLog.toArray().reduceRegion({ 
            reducer: ee.Reducer.covariance(),
            geometry: golovinGeom, 
            scale: 30, 
            maxPixels: 1e12,
            bestEffort: true,
        });

        // Extract the variances of each band (var1 and var2) and the covariance between them (covar)
        var covarMatrix = ee.Array(covariance.get('array'));
        var var1 = covarMatrix.get([0, 0]);
        var var2 = covarMatrix.get([1, 1]);
        var covar = covarMatrix.get([0, 1]);
        
        // Calculate an attenuation coefficient ratio
        var a = var1.subtract(var2).divide(covar.multiply(2));
        var attenCoeffRatio = a.add(((a.pow(2)).add(1)).sqrt());

        // Calculate a depth-invariant index for the band pair 
        // This calculation aims to normalize the effect of depth in underwater imaging
        var depthInvariantIndex = img.expression(
            'image1 - (image2 * coeff)', {
                'image1': imageLog.select([x]),
                'image2': imageLog.select([y]),
                'coeff': attenCoeffRatio
            });

        // Concatenate the DIV index for each band pair to the original image as a new band, named according to the nband list
        img = ee.Image.cat([img, depthInvariantIndex.select([x], [
            z
        ])]);
    }
    return img;
}

// Compute depth invariant indices and display
var divImg = div(kernel(makePositive(firstImage))).select ('B[2-4]', 'B2B3'); // B2B3 is used here because B2 and B3 penetrate the deepest before being absorbed
var vivVisParams = { 
    bands: ['B2B3'],
    min: -0.81,
    max: -0.04,
    gamma: 0.75
};
Map.addLayer(divImg, vivVisParams, 'divImg', false);


// Section 3: Bathymetry using Random Forests regression

//create composite image
var compositeImage = golovinBay.mosaic();

// Filter out any filler data points from DEM
function maskDEM(image) {
    var fillerMask = image.select('b1').gte(-3.4e38);
    return image.updateMask(fillerMask)
}

// Filter out the land from your DEM
function maskLandDEM(image) {
    var landMask = image.select('b1').lte(1);
    return image.updateMask(landMask)
}

var maskedDEM = maskDEM(compositeImage);
Map.addLayer(maskedDEM, {}, 'DEM no filler', false);
var maskedWaterDEM = maskLandDEM(maskedDEM);
Map.addLayer(maskedWaterDEM, {}, 'DEM no land', false);

var points = maskedWaterDEM.sample({
    region: maskedImg.geometry(), 
    scale: 1, // match the DEM resolution
    numPixels: 5000, // number of points - adjust based on your needs
    seed: 1, // seed for ensuring reproducibility
    geometries: true
});

// Split training and validation data for the bathymetry.
points = points.randomColumn();
var depthT = points.filter(ee.Filter.lte('random', 0.7)); //training data: filters out the values just created by .randomColumn to get values less than or equal to .7 (70%)
var depthV = points.filter(ee.Filter.gt('random', 0.7)); // validation data: filters out values greater than 0.7 (30%)

Map.addLayer(depthT, {
color: 'black'
}, 'Depth Training', false); 
Map.addLayer(depthV, {
color: 'gray'
}, 'Depth Validation', false); 


// Function to convert point data to a raster format for easier processing in regression analysis
function vector2image(vector) {
    // Filter out features in the vector data where the 'b1' property (representing depth) is null
    // This ensures that only valid depth measurements are included in the rasterization process
    var rasterisedVectorData = vector
        .filter(ee.Filter.neq('b1',
            null)) // Filter out Null depth values. from the rasterized image
        .reduceToImage({ 
            properties: ['b1'], // Property to include in the raster image: here, 'b1' represents depth values
            reducer: ee.Reducer.mean() // The choice of reducer depends on how you want to aggregate the data (e.g., taking the average, the sum, the maximum value, etc.).
        });
    // The resulting raster image has pixel values that represent the average depth measured at each location,
    // Providing a consistent, single measurement per pixel for use in further analysis.
    return (rasterisedVectorData); 
} 

var depthTImage = vector2image(depthT) 
.aside(Map.addLayer, {
    color: 'white'
}, 'Depth Training2', false);
var depthVImage = vector2image(depthV)
.aside(Map.addLayer, {
    color: 'white'
}, 'Depth Validation2', false);


// rfbathymetry: performs bathymetry prediction, using the Random Forest regression
function rfbathymetry(img) {
    // Sample the input image using the depth training dataset
    var training = img.sampleRegions({ 
        collection: depthT,
        scale: 3 
    });

    // Initialize a Random Forest classifier with 15 decision trees to perform regression
    var regclass = ee.Classifier.smileRandomForest(15)
        .train(training, 'b1'); // train the classifier using the 'b1'(depth) property as the label 

    // Apply the trained classifier to the image to predict bathymetric depths
    var bathyClass = img
        .classify(regclass.setOutputMode('REGRESSION')).rename(
            'b1'); 
            
    // Clip the predicted depth values to the geometry of the validation data
    // sdbEstimate will contain the depth estimate from bathyClass
    var sdbEstimate = bathyClass.clip(depthV); 


    // Concatenate estimated and validation depth data for comparison
    var imageI = ee.Image.cat([sdbEstimate, depthVImage]); //concatenates sdbEstimate and depthVImage into a single image where each pixel has two bands: one from sdbEstimate (the estimated depth from satellite data) and the other from depthVImage (the observed depth data). This operation facilitates direct comparison between the estimated and observed depths.

    // Calculate the covariance between estimated and observed depths
    var covariance = imageI.toArray().reduceRegion({
        reducer: ee.Reducer.covariance(),
        geometry: depthV,
        scale: 3, 
        bestEffort: true, 
        maxPixels: 1e9 
    });  
    var covarMatrix = ee.Array(covariance.get('array')); 

    // Convert the covariance output to an array and calculate the R² value
    var rSqr = covarMatrix.get([0, 1]).pow(2) 
        .divide(covarMatrix.get([0, 0])
            .multiply(covarMatrix.get([1, 1])));
            
    // Calculate the deviation (squared difference) between validation and estimated depths
    var deviation = depthVImage.select('mean') 
        .subtract(sdbEstimate.select('b1')).pow(2);

    // Calculate the Root Mean Square Error (RMSE) from the deviation 
    var rmse = ee.Number(deviation.reduceRegion({
            reducer: ee.Reducer.mean(),
            geometry: depthV,
            scale: 3,
            bestEffort: true,
            maxPixels: 1e12
        }).get('mean'))
        .sqrt();
        
    // Output R² and RMSE values to the console for evaluation of model performance
    print('R²', rSqr, 'RMSE', rmse); 

    // Return the classified image with predicted depth values
    return bathyClass; 
}

var clippedMask = maskedImg.clip(golovinGeom);
Map.addLayer(clippedMask, HLSvisParams, 'clipped mask');

var rfBathymetry = rfbathymetry(maskedImg); 
var clippedBathymetry = rfbathymetry(clippedMask);

var bathyVis = {
    min: -9,
    max: 1,
    palette: ['084594', '2171b5', '4292c6', '6baed6', '9ecae1', 'c6dbef', 'deebf7', 'f7fbff']// Color gradient where darker blues represent deeper areas and lighter blues represent shallower areas
};
Map.addLayer(rfBathymetry, bathyVis, 'bathymetry');
Map.addLayer(clippedBathymetry, bathyVis, 'clipped bathymetry');

//OPTIONAL: Change basemap for better visuals as the default basemap water color looks similar to the blue in the bathymetry visuals
var snazzy = require("users/aazuspan/snazzy:styles");
snazzy.addStyle("https://snazzymaps.com/style/15/subtle-grayscale", "My Custom Style");

// Section 4: visualizing the data for comparison and error analysis

// Add bathymetry and digital elevation model (DEM) layers to the map for visual comparison
Map.addLayer(clippedBathymetry, bathyVis, 'bathymetry for visualization');
Map.addLayer(maskedWaterDEM, bathyVis, 'DEM for visualization');

// Generate and print histograms for bathymetry and DEM to analyze the distribution of values
var bathyHist = ui.Chart.image.histogram(clippedBathymetry, golovinGeom, 30);
var topoHist = ui.Chart.image.histogram(maskedWaterDEM, golovinGeom, 30);
print(bathyHist, 'Bathymetry Histogram');
print(topoHist, 'Topobathy Histogram');

// Calculate and visualize the pixel-by-pixel difference between DEM and SDB images to identify potential underestimations or overestimations in the model

var differenceImage = maskedWaterDEM.subtract(rfBathymetry);
var errVisParams = {
    min: -5,
    max: 5,    
    palette: ['red', 'white', 'blue']  // Negative values in blue, positive in red
};
Map.addLayer(differenceImage, errVisParams, 'Difference Image');

// Create a histogram to analyze the distribution of differences
var differenceHistogram = ui.Chart.image.histogram(differenceImage, golovinGeom, 30);
print(differenceHistogram, 'Difference Histogram');

// Scatter plot comparing DEM and SDB to examine the correlation between predicted vs actual depths 
var image1 = maskedWaterDEM.select('b1');
var image2 = rfBathymetry.select('b1');

var sample = image1.addBands(image2) 
.sample({
    region: golovinGeom,
    scale: 30, 
    numPixels: 5000 
});

var chart = ui.Chart.feature.byFeature(sample, 'b1', 'b1_1')
.setChartType('ScatterChart')
.setOptions({
    title: 'DEM vs SDB depths',
    hAxis: {title: 'DEM (elev)'},
    vAxis: {title: 'SDB (elev)'},
    pointSize: 1,
    trendlines: {0: {}} 
});

print(chart);