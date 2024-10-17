from prefect import flow, tags

@flow
def MLbasic(userdata: str = "Train"):
    pass
if __name__ == "__main__":
    with tags("demo"):
        MLbasic()
