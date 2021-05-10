#include "Halide.h"
#include <vector>

namespace {

using namespace Halide;

Halide::Expr square(Halide::Expr e) {
  return e * e;
}

// Sorting / median methods taken from https://github.com/halide/Halide/blob/master/test/correctness/sort_exprs.cpp.
// Order a pair of Exprs, treating undefined Exprs as infinity
  void sort2(Expr &a, Expr &b) {
    if (!a.defined()) {
      std::swap(a, b);
    } else if (!b.defined()) {
      return;
    } else {
      Expr tmp = min(a, b);
      b = max(a, b);
      a = tmp;
    }
  }

// Bitonic sort a vector of Exprs
std::vector<Expr> bitonic_sort_inner(std::vector<Expr> v, bool flipped) {
  size_t size = v.size();
  size_t half_size = size / 2;

  if (!half_size) return v;

  std::vector<Expr>::iterator middle = v.begin() + half_size;
  std::vector<Expr> a, b;
  a.insert(a.begin(), v.begin(), middle);
  b.insert(b.begin(), middle, v.end());

  // Sort each half
  a = bitonic_sort_inner(a, true);
  b = bitonic_sort_inner(b, false);
  assert(a.size() == half_size);
  assert(b.size() == half_size);

  // Concat the results
  a.insert(a.end(), b.begin(), b.end());

  // Bitonic merge
  for (size_t stride = half_size; stride > 0; stride /= 2) {
    for (size_t i = 0; i < size; i++) {
      if (i % (2 * stride) < stride) {
        if (!flipped) {
          sort2(a[i], a[i + stride]);
        } else {
          sort2(a[i + stride], a[i]);
        }
      }
    }
  }

  return a;
}

std::vector<Expr> bitonic_sort(std::vector<Expr> v) {
  // Bulk up the vector to a power of two using infinities
  while (v.size() & (v.size() - 1)) {
    v.push_back(Expr());
  }

  v = bitonic_sort_inner(v, false);

  while (!v.back().defined()) {
    v.pop_back();
  }
  return v;
}

Expr median(std::vector<Expr> v) {
  v = bitonic_sort(v);
  return v[v.size() / 2];
}

class Portrait : public Halide::Generator<Portrait> {
public:
    Input<Buffer<uint8_t>> inputLeft{"inputLeft", 3};
    Input<Buffer<uint8_t>> inputRight{"inputRight", 3};
    Input<Buffer<uint8_t>> segmentedLeft{"segmentedLeft", 2};
    Output<Buffer<uint8_t>> depth_map{"depth_map", 3};
    Output<Buffer<uint8_t>> portrait{"portrait", 3};

    // Means -size ... 0 ... size.
    int matchTileSize = 3;
    // We'll search for tiles within size pixels each way of the current pixel.
    int searchWindowSize = 20;

    // TODO (rohany): It's unlikely that I will be able to get this from the dataset,
    //  and even so, I don't think it affects the final scaled outputs anyway.
    float focalLength = 5.f;
    float stereoDistance = 5.f;

    void generate() {
      // ALGORITHM.
      Var x("x"), y("y"), c("c");

      Func cInputLeft("cInputLeft"), cInputRight("cInputRight"), cSegmented("cSegmented");
      // TODO (rohany): Figure out what we can make these int sizes.
      cInputLeft(x, y, c) = cast<float_t>(Halide::BoundaryConditions::repeat_edge(inputLeft)(x, y, c));
      cInputRight(x, y, c) = cast<float_t>(Halide::BoundaryConditions::repeat_edge(inputRight)(x, y, c));
      cSegmented(x, y) = cast<float_t>(Halide::select(Halide::BoundaryConditions::repeat_edge(segmentedLeft)(x, y) < 40, 0.0f, 255.f));

      RDom tileDiffDom(
          -matchTileSize, (2 * matchTileSize) + 1, // Over the x dimension of the tile.
          -matchTileSize, (2 * matchTileSize) + 1, // Over the y dimension of the tile.
          0, 3, // Over each color channel.
          -searchWindowSize, (2 * searchWindowSize) + 1 // Over the horizontal search window.
      );

      // Doing a sum-of-squared-differences approach. Another approach is normalized correlation.
      Func tileDiff("tileDiff");
      tileDiff(x, y, tileDiffDom.w) += square(
          cInputLeft(x + tileDiffDom.x, y + tileDiffDom.y, tileDiffDom.z) -
          // We're min-reducing the tiles along the tileDiffDom.w dimension -- i.e. taking
          // all tiles within some horizontal pixel distance from the input tile.
          cInputRight(x + tileDiffDom.x + tileDiffDom.w, y + tileDiffDom.y, tileDiffDom.z)
      );

      // TODO (rohany): Comment this.
      RDom searchWindowDom(-searchWindowSize, (2 * searchWindowSize) + 1);
      Func minTile("minTile");
      minTile(x, y) = Tuple(0, std::numeric_limits<float>::max());
      Expr minx = minTile(x, y)[0];
      Expr minSS = minTile(x, y)[1];
      Expr newx = Halide::select(minSS < tileDiff(x, y, searchWindowDom.x), minx, x + searchWindowDom.x);
      Expr newMin = Halide::min(minSS, tileDiff(x, y, searchWindowDom.x));
      minTile(x, y) = Tuple(newx, newMin);

      auto infty = std::numeric_limits<float_t>::infinity();

      Func depth("depth");
      // When the disparity is 0, map the depth to infinity.
      Expr disparity = Halide::abs(x - minTile(x, y)[0]);
      depth(x, y) = Halide::select(disparity == 0,
                                   infty,
                                   // TODO (rohany): 1.f should be focalLength * stereoDistance.
                                   1.f / cast<float_t>(disparity));

      // Scale values in depth to between 0 and 255.
      RDom imageDom(0, inputLeft.width(), 0, inputLeft.height());
      Func maxDepth("maxDepth");
      maxDepth() = 0.f;
      // Map infinity to the maximum depth.
      Expr dval = depth(imageDom.x, imageDom.y);
      maxDepth() = Halide::select(dval == infty,
                                  maxDepth(),
                                  Halide::max(maxDepth(), dval));

      Func normalizedDepth("normalizedDepth");
      auto cdepth = Halide::select(depth(x, y) == infty, maxDepth(), depth(x, y));
      // This setup makes close pixels darker.
      normalizedDepth(x, y) = 255.f - (cdepth * 255.f / maxDepth());
      // This setup makes close pixels lighter.
      // normalizedDepth(x, y) = (cdepth * 255.f / maxDepth());

      // Use the segmented image to smooth the depth map. We'll do this with a median blur.
      // A gaussian blur doesn't smooth the image as much when tested.
      Func rawBlur("rawBlur");
      std::vector<Expr> exprs;
      for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
          exprs.push_back(normalizedDepth(x + i, y + j));
        }
      }
      rawBlur(x, y) = median(exprs);

      Func inMaskVals("inMaskVals"), inMaskCount("inMaskCount");
      inMaskVals() += Halide::select(cSegmented(imageDom.x, imageDom.y) == 0.f, normalizedDepth(imageDom.x, imageDom.y), 0.f);
      inMaskCount() += Halide::select(cSegmented(imageDom.x, imageDom.y) == 0.f, 1.f, 0.f);

      // Next, we'll only the blur only at positions that are in the background, as given by the segmenter.
      Func depthBlurred("depthBlurred");
      // This line makes us use the average depth for the unmasked part. This is done in the paper,
      // but it's probably fine to directly take pixels from the unmasked portion for our use case.
      // depthBlurred(x, y) = Halide::select(cSegmented(x, y) == 255.f, cast<float_t>(rawBlur(x, y)), inMaskVals() / inMaskCount());
      depthBlurred(x, y) = Halide::select(cSegmented(x, y) == 255.f, cast<float_t>(rawBlur(x, y)), normalizedDepth(x, y));

      depth_map(x, y, c) = cast<uint8_t>(depthBlurred(x, y));

      // Use the depth map to blur the image. Things further away are blurred with
      // larger blur amounts. We do this by grouping pixel depths into different groups,
      // and using a different blur size for each group.
      int numBlurLevels = 5;
      std::vector<Func> blurLevels(numBlurLevels);
      std::vector<int> cutoffs{15, 30, 45, 60, 256};
      std::vector<int> blurDimSizes{15, 12, 9, 6, 3};
      for (int i = 0; i < numBlurLevels; i++) {
        RDom dom(-blurDimSizes[i], 2 * blurDimSizes[i] + 1, -blurDimSizes[i], 2 * blurDimSizes[i] + 1);
        blurLevels[i](x, y, c) = Tuple(0.f, 0.f);
        auto val = blurLevels[i](x, y, c)[0];
        auto count = blurLevels[i](x, y, c)[1];
        // We don't include pixels from within the mask in the blur to avoid colors
        // from the foreground leaking into the background.
        Expr newVal = Halide::select(cSegmented(x + dom.x, y + dom.y) == 255.f, val + cInputLeft(x + dom.x, y + dom.y, c), val);
        Expr newCount = Halide::select(cSegmented(x + dom.x, y + dom.y) == 255.f, count + 1.f, count);
        blurLevels[i](x, y, c) = Tuple(newVal, newCount);
      }

      // Based on the value of the depth map at a pixel, choose the right blur element.
      Func backgroundBlur("backgroundBlur");
      Expr dm_val = 255.f - depth_map(x, y, c);
      backgroundBlur(x, y, c) = Halide::select(
        dm_val < cutoffs[0], blurLevels[0](x, y, c)[0] / blurLevels[0](x, y, c)[1],
        dm_val < cutoffs[1], blurLevels[1](x, y, c)[0] / blurLevels[1](x, y, c)[1],
        dm_val < cutoffs[2], blurLevels[2](x, y, c)[0] / blurLevels[2](x, y, c)[1],
        dm_val < cutoffs[3], blurLevels[3](x, y, c)[0] / blurLevels[3](x, y, c)[1],
        dm_val < cutoffs[4], blurLevels[4](x, y, c)[0] / blurLevels[4](x, y, c)[1],
        // Should never reach this case.
        0
      );

      // Add some noise back into the background of the image.
      Func syntheticNoise("synthNoise");
      syntheticNoise(x, y, c) = clamp(backgroundBlur(x, y, c) + (Halide::random_int() % 5), 0.f, 255.f);

      // For pixels that are masked out, use the blurred value. Otherwise, use the pixel
      // from the input image.
      portrait(x, y, c) = cast<uint8_t>(Halide::select(
         cSegmented(x, y) == 255.f, syntheticNoise(x, y, c), cInputLeft(x, y, c)
      ));

      // SCHEDULE.

      // Simple schedule which is way better than the default.
      tileDiff.compute_root();
      minTile.compute_root();
      depth.compute_root();
      maxDepth.compute_root();
      normalizedDepth.compute_root();
      depth_map.compute_root();
      rawBlur.compute_root();
      depthBlurred.compute_root();

      for (auto f : blurLevels) {
        f.compute_root();
      }
      backgroundBlur.compute_root();
      syntheticNoise.compute_root();

      portrait.compute_root();

      portrait.print_loop_nest();
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(Portrait, portrait_gen)
