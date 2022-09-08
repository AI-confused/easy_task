import setuptools

# with open("README.org", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="easy_task",
    version="0.0.29",
    author="black_tears",
    author_email="21860437@zju.edu.cn",
    description="make you easy to build deeplearning task",
    long_description='make you easy to build deeplearning task',
    long_description_content_type="text/markdown",
    url="https://github.com/AI-confused/easy_task",
    project_urls={
        "Bug Tracker": "https://github.com/AI-confused/easy_task/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.5",
)