from fastapi import FastAPI, Request, Depends, Form, HTTPException, Response
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from database import get_db
from models import Product, Category
from typing import List
import uuid

app = FastAPI()

# Session secret for the cart
SESSION_SECRET = "super-secret-glass-key-2026"

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Dependencies ---
def get_cart(request: Request):
    """Retrieve the cart from the session cookie."""
    cart_id = request.cookies.get("cart_id")
    if not cart_id:
        return {}

    # In a real app, we'd fetch from Redis/DB. For this demo, we'll store
    # the cart in a simple global store keyed by cart_id.
    return cart_store.get(cart_id, {})

# Simple in-memory store for carts (simulation of a database/cache)
cart_store = {}

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    featured_products = db.query(Product).filter(Product.is_featured == True).all()
    categories = db.query(Category).all()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "featured_products": featured_products,
        "categories": categories
    })

@app.get("/category/{slug}", response_class=HTMLResponse)
async def category_page(request: Request, slug: str, db: Session = Depends(get_db)):
    category = db.query(Category).filter(Category.slug == slug).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    products = db.query(Product).filter(Product.category_id == category.id).all()
    return templates.TemplateResponse("category.html", {
        "request": request,
        "category": category,
        "products": products
    })

@app.get("/product/{slug}", response_class=HTMLResponse)
async def product_page(request: Request, slug: str, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.slug == slug).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    return templates.TemplateResponse("product_detail.html", {
        "request": request,
        "product": product
    })

@app.post("/cart/add/{product_id}")
async def add_to_cart(request: Request, response: Response, product_id: int):
    cart_id = request.cookies.get("cart_id")
    if not cart_id:
        cart_id = str(uuid.uuid4())
        response.set_cookie(key="cart_id", value=cart_id)

    cart = cart_store.get(cart_id, {})
    cart[str(product_id)] = cart.get(str(product_id), 0) + 1
    cart_store[cart_id] = cart

    return RedirectResponse(url="/cart", status_code=303)

@app.get("/cart", response_class=HTMLResponse)
async def cart_page(request: Request, db: Session = Depends(get_db)):
    cart_id = request.cookies.get("cart_id")
    cart = cart_store.get(cart_id, {}) if cart_id else {}

    cart_items = []
    total = 0
    for p_id, qty in cart.items():
        product = db.query(Product).filter(Product.id == int(p_id)).first()
        if product:
            subtotal = product.price * qty
            total += subtotal
            cart_items.append({"product": product, "quantity": qty, "subtotal": subtotal})

    return templates.TemplateResponse("cart.html", {
        "request": request,
        "items": cart_items,
        "total": total
    })

@app.post("/cart/remove/{product_id}")
async def remove_from_cart(request: Request, product_id: int):
    cart_id = request.cookies.get("cart_id")
    if cart_id and cart_id in cart_store:
        cart = cart_store[cart_id]
        if str(product_id) in cart:
            del cart[str(product_id)]

    return RedirectResponse(url="/cart", status_code=303)

@app.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

@app.post("/contact/send", response_class=HTMLResponse)
async def send_contact(request: Request):
    form_data = await request.form()
    # Here you would typically send an email or save to DB
    # For the demo, we just redirect to success.
    return templates.TemplateResponse("success.html", {"request": request, "name": form_data.get("name")})

@app.get("/admin/seed")
async def trigger_seed():
    import seed_db
    seed_db.seed_data()
    return {"message": "Database seeded successfully"}
