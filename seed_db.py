from database import SessionLocal, engine, init_db
from models import Category, Product

def seed_data():
    # Ensure tables exist
    init_db()

    db = SessionLocal()
    try:
        # Check if already seeded
        if db.query(Category).first():
            print("Database already seeded.")
            return

        print("Seeding database...")

        # Categories
        categories_data = [
            {"name": "Hand-Blown Luxury", "slug": "hand-blown", "description": "Artisanal pieces crafted by master glassblowers."},
            {"name": "Crystal Collection", "slug": "crystal", "description": "Precision-cut high-lead crystal for elegance."},
            {"name": "Modern Minimalist", "slug": "modern", "description": "Sleek lines and functional beauty for the modern home."}
        ]

        categories = []
        for cat in categories_data:
            category = Category(**cat)
            db.add(category)
            categories.append(category)

        db.commit()
        # Refresh to get IDs
        for cat in categories:
            db.refresh(cat)

        # Products
        products_data = [
            {
                "name": "Azure Dream Goblet",
                "slug": "azure-dream-goblet",
                "description": "A breathtaking hand-blown goblet featuring deep azure swirls. Perfect for special occasions.",
                "price": 120.0,
                "stock": 15,
                "image_url": "https://images.unsplash.com/photo-1516626951303-6273f4e33730?auto=format&fit=crop&w=800&q=80",
                "is_featured": True,
                "category_id": categories[0].id
            },
            {
                "name": "Diamond Crystal Flute",
                "slug": "diamond-crystal-flute",
                "description": "Exceptional clarity and precision cuts. Designed to keep champagne sparkling longer.",
                "price": 85.0,
                "stock": 30,
                "image_url": "https://images.unsplash.com/photo-1550985506-7867a5179703?auto=format&fit=crop&w=800&q=80",
                "is_featured": True,
                "category_id": categories[1].id
            },
            {
                "name": "Nordic Mist Tumbler",
                "slug": "nordic-mist-tumbler",
                "description": "Minimalist aesthetic with a subtle frosted finish. A staple for contemporary dining.",
                "price": 45.0,
                "stock": 100,
                "image_url": "https://images.unsplash.com/photo-1515696952069-50837767676e?auto=format&fit=crop&w=800&q=80",
                "is_featured": False,
                "category_id": categories[2].id
            },
            {
                "name": "Golden Hour Decanter",
                "slug": "golden-hour-decanter",
                "description": "A masterpiece of balance and light. Hand-blown with gold-leaf accents.",
                "price": 250.0,
                "stock": 5,
                "image_url": "https://images.unsplash.com/photo-1584739139618-777f21bb391e?auto=format&fit=crop&w=800&q=80",
                "is_featured": True,
                "category_id": categories[0].id
            },
            {
                "name": "Celestial Sphere Vase",
                "slug": "celestial-sphere-vase",
                "description": "An ethereal centerpiece capturing the essence of the night sky in glass.",
                "price": 180.0,
                "stock": 10,
                "image_url": "https://images.unsplash.com/photo-1612547848777-1a7a1f77982d?auto=format&fit=crop&w=800&q=80",
                "is_featured": False,
                "category_id": categories[0].id
            },
            {
                "name": "Pure Clarity Water Glass",
                "slug": "pure-clarity-water-glass",
                "description": "Ultra-clear glass with a balanced weight. Simple, elegant, and timeless.",
                "price": 30.0,
                "stock": 200,
                "image_url": "https://images.unsplash.com/photo-1574943321760-657b89535418?auto=format&fit=crop&w=800&q=80",
                "is_featured": False,
                "category_id": categories[2].id
            }
        ]

        for prod in products_data:
            product = Product(**prod)
            db.add(product)

        db.commit()
        print("Successfully seeded the glass platform database!")

    except Exception as e:
        print(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_data()
