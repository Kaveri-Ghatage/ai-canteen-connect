import mongoose from "mongoose";

// The connection string comes from backend/.env (never commit it):
//   MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/food-del
export const connectDB = async () => {
    if (!process.env.MONGODB_URI) {
        console.error("MONGODB_URI is not set. Add it to backend/.env");
        process.exit(1);
    }
    await mongoose.connect(process.env.MONGODB_URI).then(() => console.log("DB Connected"));
};
