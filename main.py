import stereo
import semantic_segmentation
import locational
import reconstruction

if __name__ == '__main__':
    group = 6
    stereo.stereo_rect(group)
    # semantic_segmentation.seg()
    locational.output_model(group)
    reconstruction.main()
