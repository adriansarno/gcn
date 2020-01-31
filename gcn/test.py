    

# -------------------------------------------------------------------------
def main():

    # Plot the annotations and tags in one image to verify that the entity-word matching function works properly.

    # find files
    image_files, words_files = get_raw_filepaths(task1train_dir)
    _, entities_files = get_raw_filepaths(task2train_dir)

    # load one raw example
    image, word_areas, word2entity = load_raw_example(image_files[0], words_files[0], entities_files[0])

    # match entity tags to words (approximately) and covert entities to tabular format
    entities = match_words_to_entities(word_areas, word2entity)

    # create an image with annotations
    plot_image(image, word_areas, entities)

    print("Visualize Raw Data")
    print("writing {}".format("raw_example.png"))
    plt.savefig('raw_example.png')

if __name__ == "__main__":
    main()