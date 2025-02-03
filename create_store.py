import os
import ast
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
# from utils import load_vectorstore
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
import os
# Define the metadata extraction function.
load_dotenv()



def get_all_products(shop_url, api_version):
    all_products = []
    url = f"{shop_url}/admin/api/{api_version}/products.json"
    headers = {"X-Shopify-Access-Token": os.getenv("SHOPIFY_API_KEY")}
    params = {"limit": 250}
    response = requests.get(url, headers=headers, params=params)
    all_products.extend(response.json()["products"])
    try:
        while response.links["next"]:
            response = requests.get(response.links["next"]["url"], headers=headers)
            all_products.extend(response.json()["products"])
            # time.sleep(2)
            print("total_products:", len(all_products))
    except KeyError:
        return all_products
    

def clean_data(all_products):
    
    df = pd.DataFrame(all_products)
    # df = df[~df['status'].isin(['archived', 'draft'])]
    df = df[df["status"]=="active"]
    df.fillna("",inplace=True)
    

    # Function to clean HTML tags from the 'body_html' column
    def clean_html_tags(row):
        html_content = row["body_html"]  # Access the specific column
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text()
        row["body_html"] = text  # Update the column in the row
        return row
    
    def get_img_src(row):
        img_list = row["images"]
        if isinstance(img_list, list):
            # Extract all "src" attributes from the list of dictionaries
            img_src_list = [img["src"] for img in img_list if "src" in img]
            row["image_list"] = img_src_list  # Store as a list
        else:
            row["image_list"] = []  # Handle cases where no images are found
        return row
    
    def generate_link(row):
        if pd.notna(row['handle']):  # Check if 'handle' is not NaN
            row["link"] = f"https://jacksonshardware.myshopify.com/products/{row['handle']}"
        else:
            row["link"] = ""  # Empty string or any placeholder you prefer if 'handle' is missing
        return row
    def extract_price_and_inventory_quantity(row):
        # Parse the 'variants' column
        if isinstance(row['variants'], str):
            row['variants'] = ast.literal_eval(row['variants'])
        
        # Extract 'price' and 'inventory_quantity'
        if "variants" in row:
            variant = row["variants"][0]
            row["price"] = variant["price"]
            row["inventory_quantity"] = variant["inventory_quantity"]
        else:
            row["price"] = None
            row["inventory_quantity"] = None
        return row

    def create_expanded_descriptions(row):
        row["expanded_description"] = ""
        
        # Check and add the 'title' field if it's not empty
        if row['title'] != "":
            row["expanded_description"] += "Title: " + row['title'] + " "
        
        # Check and add the 'description' field if it's not empty
        if row['description'] != "":
            row["expanded_description"] += "Description: " + row['description'] + " "
        
        # Check and add the 'vendor' field if it's not empty
        if row['vendor'] != "":
            row["expanded_description"] += "Vendor: " + row['vendor'] + " "
        
        # Check and add the 'product_type' field if it's not empty
        if row['product_type'] != "":
            row["expanded_description"] += "Product Type: " + row['product_type'] + " "
        
        # Check and add the 'tags' field if it's not empty
        if row['tags'] != "":
            row["expanded_description"] += "Tags: " + row['tags'] + " "
        
        return row
    

    # def df_preprocessing(df):
   
    #     return df
    df = df.apply(clean_html_tags, axis=1)
    df = df.apply(get_img_src, axis=1)
    df = df.rename(columns={"body_html": "description"})
    df = df.apply(create_expanded_descriptions, axis=1)
    df = df.apply(generate_link, axis=1)
    df = df.apply(extract_price_and_inventory_quantity, axis=1)
    df = df[["id","title","description","expanded_description","image_list","tags","link","price","inventory_quantity","vendor","product_type"]]
             
    cleaned_df = df

    cleaned_products_json = cleaned_df.to_json(orient='records')
    with open("products.json","w") as f:
        f.write(cleaned_products_json)

    cleaned_df.to_csv("products.csv", index=False)


    return


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["id"] = record.get("id")
    metadata["title"] = record.get("title")
    metadata["tags"] = record.get("tags")
    metadata["description"] = record.get("description")
    metadata["image_list"] = record.get("image_list")
    metadata["price"] = record.get("price")
    metadata["inventory_quantity"] = record.get("inventory_quantity")
    metadata["vendor"] = record.get("vendor")
    metadata["product_type"] = record.get("product_type")
    metadata["link"] = record.get("link")
    return metadata

def create_vectorstore(documents,embeddings):
    vectorstore = FAISS.from_documents(documents=documents,embedding=embeddings)
    return vectorstore
 
def save_vectorstore(vectorstore,save_path,index_name):
    vectorstore.save_local(save_path,index_name)
    print("vectorstore saved to : ", save_path)
    return None

# def load_vectorstore(vectorstore_path,index_name):
#     vectorstore = FAISS.load_local(folder_path = vectorstore_path,index_name = index_name, embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") , allow_dangerous_deserialization=True)
#     return vectorstore



def open_vectorstore():
    shop_url = os.getenv('shop_url')
    api_version = os.getenv('api_version')
    all_products = get_all_products(shop_url,api_version)
    print("data collected...")
    clean_data(all_products)
    print("data cleaned...")


    loader = JSONLoader(
    file_path='./products.json',
    jq_schema='.[]',
    content_key="expanded_description",
    metadata_func=metadata_func
    )
        
    documents = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = create_vectorstore(documents,embeddings)
    save_vectorstore(vectorstore,save_path = "shopify_langchain_testing_vectorstore",index_name = "products")
    # vectorstore = load_vectorstore(vectorstore_path="shopify_langchain_testing_vectorstore",index_name="products")
    # print("vectorstore loaded successfully")

if __name__ == "__main__":
    open_vectorstore()
