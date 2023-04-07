# Define input and output file patterns
input_pattern = "data/{subfolders}/{filename}.lif"
output_pattern = "data/{subfolders}/{filename}.zarr"


# Define all rule to generate all .zarr files from corresponding .czi files
rule all:
    input:
        expand(
            output_pattern,
            zip,
            subfolders=glob_wildcards(input_pattern).subfolders,
            filename=glob_wildcards(input_pattern).filename,
        ),
# Define rule to convert .czi files to .zarr using bioformats2raw
rule convert_to_zarr:
    input:
        "data/{subfolders}/{filename}.lif"
    output:
        directory("data/{subfolders}/{filename}.zarr")
    shell:
        """
        bioformats2raw '{input}' '{output}'
        """


